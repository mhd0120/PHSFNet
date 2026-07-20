"""Train OCAP-Net on prepared OCO-2 parquet files.

Default split:
    Train: 2020-2021
    Validation / fine-tuning source: 2022
    Test: 2023
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from xco2net.model import OCAPGateNet
from xco2net.process_data import read_data, build_scaler, create_dataset
from xco2net.save import Save
from xco2net.train import EarlyStopper, TrainInClass


def seed_everything(seed=1237):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train OCAP-Net for OCO-2 XCO2 residual estimation.")
    parser.add_argument("--seed", type=int, default=1237)
    parser.add_argument("--tag", type=str, default="ocap")
    parser.set_defaults(finetune=True)
    parser.add_argument("--finetune", dest="finetune", action="store_true", help="Enable two-stage fine-tuning.")
    parser.add_argument("--no_finetune", dest="finetune", action="store_false", help="Disable two-stage fine-tuning.")
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--gate_rescale", dest="gate_rescale", action="store_true", default=True)
    parser.add_argument("--no_gate_rescale", dest="gate_rescale", action="store_false")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_save_name = f"OCAPNet_finetune_{args.tag}" if args.finetune else f"OCAPNet_{args.tag}"
    train_log_dir = f"runs_xco2/{model_save_name}"

    features_names = [
        "relative_azimuth", "solar_zenith", "zenith", "polarization_angle",
        "solar_distance", "relative_velocity", "sounding_index",
        "surface_pressure_apriori_fph", "xco2_apriori",
    ]
    band_names = ["o2", "weak_co2", "strong_co2"]
    array_names = ["radiances", "snrs"]
    label_names = ["xco2_diff"]
    error_scale_names = ["xco2_uncert"]
    column_names = [features_names, band_names, array_names, label_names, error_scale_names]

    print("Reading year-split data...")
    df_2020_2021 = read_data(["2020", "2021"])
    df_2022 = read_data(["2022"])
    df_2023 = read_data(["2023"])

    label_scaler, feature_scaler, spectrum_scalers, error_scaler = build_scaler(df_2020_2021, *column_names)
    scalers = [label_scaler, feature_scaler, spectrum_scalers, error_scaler]

    save = Save(model_save_name)
    save.save_columns_names(features_names, band_names, array_names, label_names, error_scale_names)
    save.save_scaler(*scalers)

    if args.finetune:
        split_pt = len(df_2022) // 2
        df_ft_train = df_2022.iloc[:split_pt]
        df_ft_valid = df_2022.iloc[split_pt:]

        pretrain_train = create_dataset(df_2020_2021, *scalers, *column_names)
        pretrain_valid = create_dataset(df_ft_train, *scalers, *column_names)
        pretrain_test = create_dataset(df_2023, *scalers, *column_names)
        datasets = [pretrain_train, pretrain_valid, pretrain_test]

        ft_train = create_dataset(df_ft_train, *scalers, *column_names)
        ft_valid = create_dataset(df_ft_valid, *scalers, *column_names)
        ft_test = create_dataset(df_2023, *scalers, *column_names)
        ft_datasets = [ft_train, ft_valid, ft_test]
    else:
        train_dataset = create_dataset(df_2020_2021, *scalers, *column_names)
        valid_dataset = create_dataset(df_2022, *scalers, *column_names)
        test_dataset = create_dataset(df_2023, *scalers, *column_names)
        datasets = [train_dataset, valid_dataset, test_dataset]
        ft_datasets = None

    d_model = 256
    feature_dim = datasets[0][0]["features"].shape[0]
    config = {
        "patch_size": 1016,
        "stride": 1016,
        "feature_dim": feature_dim,
        "d_model": d_model,
        "final_mlp_layers": [128, 1],
        "use_state_band_gate": True,
        "gate_hidden_dim": args.gate_hidden_dim,
        "gate_temperature": args.gate_temperature,
        "gate_rescale": args.gate_rescale,
    }
    save.save_config(config)

    label_info = [label_scaler.mean_, label_scaler.scale_]
    criterion = nn.HuberLoss(delta=0.5)
    model = OCAPGateNet(**config).to(device)

    print("Stage 1: pretraining")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs, eta_min=1e-7)
    early_stopper = EarlyStopper(patience=50, min_delta=1e-6)
    trainer = TrainInClass(
        train_log_dir, model_save_name, model, criterion, optimizer, scheduler,
        epochs=args.pretrain_epochs, batch_size=args.batch_size,
        early_stopper=early_stopper, device=device, datas=datasets,
        stad_info=label_info, feature_scaler=feature_scaler,
    )
    trainer.train()
    model.load_state_dict(torch.load(f"model_status/{model_save_name}/weight/best.pth", map_location=device))
    fig = trainer.make_plot(100000)
    fig.savefig(f"model_status/{model_save_name}/img/pretrain.png")

    if args.finetune:
        print("Stage 2: fine-tuning")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-7)
        early_stopper = EarlyStopper(patience=30, min_delta=1e-6)
        trainer_ft = TrainInClass(
            train_log_dir, model_save_name, model, criterion, optimizer, scheduler,
            epochs=args.finetune_epochs, batch_size=args.batch_size,
            early_stopper=early_stopper, device=device, datas=ft_datasets,
            stad_info=label_info, feature_scaler=feature_scaler,
        )
        trainer_ft.train()
        torch.save(model.state_dict(), f"model_status/{model_save_name}/weight/best.pth")
        fig = trainer_ft.make_plot(100000)
        fig.savefig(f"model_status/{model_save_name}/img/finetune.png")

    final_fig = trainer.make_plot(100000)
    final_fig.savefig(f"model_status/{model_save_name}/img/final.png")
    print(f"Done. Best model saved in model_status/{model_save_name}/weight/best.pth")


if __name__ == "__main__":
    main()

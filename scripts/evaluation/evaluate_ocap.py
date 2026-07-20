"""Evaluate a trained OCAP-Net checkpoint on the 2023 test split."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xco2net.model import OCAPGateNet
from xco2net.process_data import read_data, create_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True, help="Directory under model_status or an absolute path.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "ocap_test_predictions.csv")
    return parser.parse_args()


def resolve_model_dir(path):
    if path.is_absolute():
        return path
    direct = PROJECT_ROOT / path
    if direct.exists():
        return direct
    return PROJECT_ROOT / "model_status" / path


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    weight_path = model_dir / "weight" / "best.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (model_dir / "column_names.json").open("r", encoding="utf-8") as f:
        columns = json.load(f)

    label_scaler = joblib.load(model_dir / "label_scaler.pkl")
    feature_scaler = joblib.load(model_dir / "feature_scaler.pkl")
    spectrum_scalers = joblib.load(model_dir / "spectrum_scalers.pkl")
    error_scaler = joblib.load(model_dir / "error_scaler.pkl")

    df_test = read_data(["2023"])
    dataset = create_dataset(
        df_test,
        label_scaler,
        feature_scaler,
        spectrum_scalers,
        error_scaler,
        columns["features_names"],
        columns["band_names"],
        columns["array_names"],
        columns["label_names"],
        columns["error_scale_names"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCAPGateNet(**config).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    y_true_scaled = []
    y_pred_scaled = []
    gates = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            features = batch["features"].to(device)
            spectra = batch["spectra"].to(device)
            labels = batch["labels"].cpu().numpy().ravel()
            output, aux = model(features, spectra, return_aux=True)
            y_true_scaled.append(labels)
            y_pred_scaled.append(output.cpu().numpy().ravel())
            if "band_gates" in aux:
                gates.append(aux["band_gates"].cpu().numpy())

    y_true_scaled = np.concatenate(y_true_scaled)
    y_pred_scaled = np.concatenate(y_pred_scaled)
    label_mean = float(label_scaler.mean_[0])
    label_std = float(label_scaler.scale_[0])
    xco2_apriori = df_test[("features", "xco2_apriori")].values.ravel()

    y_true = (y_true_scaled * label_std + label_mean + xco2_apriori) * 1e6
    y_pred = (y_pred_scaled * label_std + label_mean + xco2_apriori) * 1e6

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE={mae:.4f} ppm, RMSE={rmse:.4f} ppm, R2={r2:.4f}")

    out = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
    })
    if gates:
        gates = np.concatenate(gates, axis=0)
        out["gate_o2"] = gates[:, 0]
        out["gate_wco2"] = gates[:, 1]
        out["gate_sco2"] = gates[:, 2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

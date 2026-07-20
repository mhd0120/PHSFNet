# OCAP-Net

OCAP-Net (Observation-Conditioned Absorption Pattern Network) is a neural model
for estimating column-averaged dry-air CO2 mole fraction (XCO2) from OCO-2
spectral observations.

The model uses:

- wavelength-aligned SNR-aware radiance weighting;
- band-wise spectral encoding for the O2 A-Band, weak CO2 band, and strong CO2 band;
- state-conditioned band-level gates generated from auxiliary physical states;
- residual learning, where the model predicts `xco2 - xco2_apriori`.

## Repository Layout

```text
OCAP-main/
  train_the_model.py              # main OCAP-Net training script
  xco2net/
    model.py                      # OCAP-Net model only
    process_data.py               # parquet loading, scaling, dataset creation
    train.py                      # training loop, prediction, gate logging
    save.py                       # saves configs, scalers, checkpoints
    support_func.py               # small metric helpers
  scripts/
    data_preparation/
      organize_data.py            # optional raw OCO-2 L1/L2 to parquet utility
    evaluation/
      evaluate_ocap.py            # evaluate trained OCAP checkpoint
    plotting/                     # optional plotting scripts
  fast_read/full/                 # put processed parquet files here
  data/                           # raw data placeholder; not tracked
  model_status/                   # training checkpoints; not tracked
  outputs/                        # logs and gate statistics; not tracked
  results/                        # evaluation tables and figures; not tracked
```

## Data

The raw OCO-2 products can be downloaded from NASA's OCO-2 data portal:

- OCO-2 data overview: https://ocov2.jpl.nasa.gov/science/oco-2-data-center/
- NASA Earthdata Search: https://search.earthdata.nasa.gov/

This project expects processed yearly parquet files:

```text
fast_read/full/training_data_2020.parquet
fast_read/full/training_data_2021.parquet
fast_read/full/training_data_2022.parquet
fast_read/full/training_data_2023.parquet
```

The training split is fixed as:

- Train: 2020-2021
- Validation / fine-tuning source: 2022
- Test: 2023

Large data files are intentionally not included in this repository. Keep raw
OCO-2 files under `data/` or another local directory, then place the processed
parquet files under `fast_read/full/`.

## Installation

```bash
conda create -n ocap python=3.10
conda activate ocap
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build if GPU training is available.

## Training

Default training uses two stages: pretraining on 2020-2021 with validation from
half of 2022, then fine-tuning on 2022.

```bash
python train_the_model.py --seed 1237 --tag ocap
```

Disable fine-tuning:

```bash
python train_the_model.py --no_finetune --seed 1237 --tag ocap_no_ft
```

Gate parameters can be changed from the command line:

```bash
python train_the_model.py \
  --seed 1237 \
  --tag ocap_gate_t1 \
  --gate_hidden_dim 64 \
  --gate_temperature 1.0 \
  --gate_rescale
```

Outputs are saved under:

```text
model_status/OCAPNet_finetune_<tag>/
outputs/OCAPNet_finetune_<tag>/
runs_xco2/OCAPNet_finetune_<tag>/
```

The best checkpoint is:

```text
model_status/OCAPNet_finetune_<tag>/weight/best.pth
```

## Evaluation

Evaluate a trained checkpoint on the 2023 test split:

```bash
python scripts/evaluation/evaluate_ocap.py \
  --model_dir model_status/OCAPNet_finetune_ocap \
  --output results/ocap_test_predictions.csv
```

Metrics are computed on restored XCO2 values:

```text
y_hat = xco2_apriori + predicted_residual
```

The evaluation CSV also contains the learned band gates when available.

## Data Preparation Notes

`scripts/data_preparation/organize_data.py` is a utility for converting raw
OCO-2 Level 1/Level 2 files into parquet tables. It assumes local index JSON
files mapping orbits to raw OCO-2 product paths. In practice, users often adapt
this script to their local Earthdata download layout before generating the final
`training_data_<year>.parquet` files.

## Citation

If this code is used in a paper, please cite the associated OCAP-Net manuscript
once available.

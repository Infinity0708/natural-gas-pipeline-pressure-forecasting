# Natural Gas Pipeline Forecasting Project

## Overview
This repository contains an end-to-end forecasting workflow for natural gas pipeline pressure modelling using industrial operational time series. The project integrates data processing, feature engineering, model development, hyperparameter optimisation, evaluation, and experiment tracking.

The main objective is to forecast short-horizon station pressure from multivariate hourly operational data, including throughput-related variables, station pressures, temperatures, fuel-gas consumption, and compressor-state indicators. The workflow also documents both successful and unsuccessful modelling iterations, with a strong emphasis on reproducibility.

## Main Contributions
Built a reusable data-processing pipeline to convert raw station records into a model-ready hourly table.
Implemented and compared five forecasting model families:
 Ridge Regression
 CNN1D
 LSTM
 BiLSTM
 CNN-BiLSTM
Added Optuna-based hyperparameter optimisation (HPO) for all major neural models.
Introduced **delta-target learning** for robust pressure forecasting under chronological train/validation/test splits.
Standardised model outputs: each training run exports a checkpoint, JSON report, forecast curve, loss curve, and test prediction CSV.

## Objective
The goal of the project is to build predictive models that accurately forecast station-level pressure from real operational data. The resulting models support:
 short-term pressure forecasting,
 model comparison under a common protocol,
 future integration with compressor scheduling and energy-efficiency analysis.

## Methodology Summary
1. Data integration
    Consolidate raw CSV and spreadsheet-derived references.
    Parse timestamps and align multivariate station signals.
    Construct a single hourly dataset for supervised learning.

2. Feature engineering
    Build station and line level features (pressure, temperature, fuel-gas consumption, compressor state).
    Add optional target-lag features.
    Standardise inputs and targets using training-set statistics only.

3. Forecasting formulation
    Sliding-window supervised learning with configurable lookback and forecast horizon.
    Chronological split: 70% train / 10% validation / 20% test.
    Two target modes:
      `abs`: direct future pressure
      `delta`: future pressure change, reconstructed to absolute pressure after prediction

4. Models
    Ridge Regression for linear baseline.
    CNN1D for local temporal feature extraction.
    LSTM and BiLSTM for recurrent temporal modelling.
    CNN-BiLSTM for hybrid local/global sequence learning.

5. Hyperparameter optimisation
    Optuna-based search over model-specific spaces.
    Best configurations saved as JSON.
    Final train scripts re-run the selected best parameters for reproducible reporting.

6. Evaluation
    Metrics: MAE, RMSE, R^2
    Final reported metrics are interpreted in the original physical scale after inverse transformation and delta reconstruction.

## Repository Structure
xxc262/
│
│
├── hpo/
│   ├── hpo_utils.py
│   ├── lstm_auto_hpo.py
│   ├── model1_ridge_hpo.py
│   ├── model2_cnn1d_hpo.py
│   ├── model3_bilstm_hpo.py
│   ├── model4_lstm_hpo.py
│   └── model5_cnn_bilstm_hpo.py
│
├── raw/
│   ├── raw.csv
│   ├── day_dataset.csv
│   ├── 数据处理结果.csv
│   └── 需处理点位表.csv
│
├── runs/
│   ├── lstm/
│   ├── model1_ridge/
│   ├── model2_cnn1d/
│   ├── model3_bilstm/
│   ├── model4_lstm/
│   └── model5_cnn_bilstm/
│
├── src/
│   ├── build_day_dataset.py
│   ├── process_data(1).py
│   ├── model1_ridge_train.py
│   ├── model2_cnn1d_train.py
│   ├── model3_bilstm_train.py
│   ├── model4_lstm_train.py
│   ├── model5_cnn_bilstm_train.py
│   ├── lstm_model_training.py
│   ├── lstm-model-training-westline1.py
│   ├── stgnn_forecast.py
│   ├── stgnn_best.pt
│   ├── best_model.pt
│   ├── lstm_fixed_torch.pt
│   ├── lstm_fixed_torch_metadata.pkl
│   ├── optimized_lstm_torch.pt
│   └── optimized_lstm_torch_metadata.pkl
│
├── venv/
├── README.md
├── logbook.md
└── .gitignore

## Important Experimental Findings
 Absolute-target neural forecasting can fail under strict chronological evaluation.
 Delta-target forecasting is much more robust and became the default setting for neural models.
 CNN1D, LSTM, BiLSTM, and CNN-BiLSTM reached similar short-horizon RMSE once delta mode and target lag were enabled.
 A later BiLSTM refinement improved validation error but degraded test performance, and is retained as a documented negative result.

## Typical Run Flow
1. Prepare the processed dataset in `raw/数据处理结果.csv`
2. Run HPO:
   python hpo/modelX_xxx_hpo.py ...
3. Train final model with best parameters:
   python src/modelX_xxx_train.py ...
4. Review outputs under `runs/modelX_xxx/...`

## Future Work
 Walk-forward validation under changing operating regimes.
 Transfer learning across different pipeline lines.
 Multi-step direct forecasting for 1-10 hour horizons.
 Integration with downstream operational optimisation.

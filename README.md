# Natural Gas Pipeline Prediction Modelling
  Overview:
This project focuses on natural gas pipeline flow and energy consumption forecasting using multiple machine learning and deep learning models.
It provides a complete pipeline including:
    Data preprocessing and feature engineering
   Daily dataset construction
   Model training (Ridge, CNN1D, LSTM, BiLSTM, STGNN，CNN-BiLSTM)
   Hyperparameter optimization (HPO with Optuna)
   Model evaluation and result storage
The project is designed for industrial forecasting scenarios, such as pipeline transport prediction, energy consumption estimation, and operational optimization.

##  Objective
The objective of this project is to build a predictive model that accurately forecasts the inlet pressure or energy consumption of natural gas compressor stations based on real operational data.  
By analyzing multi-station time series data (pressure, temperature, flow rate, etc.), the model helps identify deviations between predicted and actual energy use, supporting intelligent optimization of compressor operations.


## Methodology

1. Data Integration
   - Merge and clean time series data from multiple compressor stations.  
   - Align timestamps and unify measurement points (inlet/outlet pressure, temperature, flow).  

2. Feature Engineering
   - Generate temporal features (`hour`, `day_of_week`, etc.) and lag variables.  
   - Apply normalization using `MinMaxScaler`.

3. Model Development 
   - Build a multi-layer LSTM network (PyTorch) for time series forecasting.  
   - Implement early stopping and validation-based checkpoint saving.  
   - Train using sliding windows of past 30 time steps to predict the next value.

4. Model Evaluation
   - Metrics: RMSE, MAE, R²  
   - Visualization: time-series comparison, parity plots, residual analysis.

5. Future Work 
   - Extend to multi-station joint prediction.  
   - Integrate energy optimization and anomaly detection modules.


## Project Structure
xxc262/
│
├── figs/                     
│
├── hpo/                      
│   ├── hpo_utils.py
│   ├── lstm_auto_hpo.py
│   ├── model1_ridge_hpo.py
│   ├── model2_cnn1d_hpo.py
│   └── model3_bilstm_hpo.py
│
├── raw/                      
│   ├── raw.csv
│   ├── day_dataset.csv
│   ├── processed_data.csv
│   ├── pipeline_model.xlsx
│   ├── energy_model.xlsx
│   └── indicators_doc.xlsx
│
├── run/ 
│
├── runs/                     
│   ├── lstm/
│   ├── model1_ridge/
│   ├── model2_cnn1d/
│   └── model3_bilstm/
│
├── src/                      
│   │
│   ├── Data Processing
│   │   ├── process_data.py
│   │   └── build_day_dataset.py
│   │
│   ├── Model Training
│   │   ├── model1_ridge_train.py
│   │   ├── model2_cnn1d_train.py
│   │   ├── model3_bilstm_train.py
│   │   ├── lstm_model_training.py
│   │   └── lstm-model-training-westline1.py
│   │
│   ├── STGNN
│   │   ├── stgnn_forecast.py
│   │   └── stgnn_best.pt
│   │
│   ├── Saved Models
│   │   ├── best_model.pt
│   │   ├── optimized_lstm_torch.pt
│   │   └── *.pkl metadata
│   │
│   └── Visualization
│       └── fig_train_curve_*.png
│
├── venv/                     
├── README.md
├── logbook.md        
└── .gitignore
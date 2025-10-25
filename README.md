# Natural Gas Pipeline Prediction Modelling

This project focuses on developing a data-driven fatigue (energy consumption) modelling and prediction system for natural gas compressor stations.  
It integrates LSTM-based deep learning models with domain knowledge from pipeline operations to forecast station inlet/outlet pressures and optimize compressor efficiency.


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


##  Project Structure

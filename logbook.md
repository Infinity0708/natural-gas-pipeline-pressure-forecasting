## Natural Gas Pipline Prediction Modelling Logbook

### 18 Oct 2025
Task: Finish data collecting.

Outcome: Completed the integration of West Line 2 and West Line 3 parameters into raw.csv

### 20 Oct 2025
Task: Finish data cleaning and Complete data cleaning, feature classification, and data set classification.

Outcome: Finish the code in lstm-model-training.py 

### 23 Oct 2025
Task: Meeting with supervisor and communicate next project steps.

Outcome: Read the model report or essay about time series, and run the test data with the basic LSTM model first.

### 2 Nov 2025
Task: Start Working on the next step use LSTM basic model.

Outcome: The debugging code is expected to be completed in the first week of November.

### 3 Nov 2025
Task: Define the LSTM-based regression model for time series prediction.

Outcome: Update the code in lstm-model-training.py

### 4 Nov 2025
Task: Complete the time series functions and the prediction of loss function.

Outcome: Start working on the time series issue.

### 5 Nov 2025
Task: Finish the first version of LSTM prediction model and debug the code and finally it can execute.
Also, add loss function to analyze the difference between the image judgement model's predictions and actual data.

Outcome: Convert the time series into supervised sliding-window samples, train a 2-layer LSTM regressor with strict time
splits and separate scalers to avoid leakage, use early stopping and LR scheduling for stable training, evaluate in original 
units (MPa) with RMSE/MAE/R², and save weights plus metadata for reproducible deployment

### 10 Nov 2025
Code Interpretation:
Batch size is the number of samples processed in one forward–backward pass.
It controls memory usage, speed, and the noise level of the gradient:
Larger batches → higher memory, fewer steps per epoch, smoother gradients, potentially larger learning rate. 
Smaller batches → lower memory, noisier gradients (sometimes better generalization).

### 13 Nov 2025
Code Interpretation:
Epoch: one full pass over the training dataset.
In each epoch, the model processes all mini-batches, updates parameters,
then we evaluate on the validation set.
We use ReduceLROnPlateau to adjust the learning rate per epoch and early stopping to halt training when validation loss no longer improves.
When we get the lowest val_loss is saved as best_model.pt.

Why we have many epoch?
Because it's necessary to tranverse the training set multiple times and continuosly adjust the parameters to achieve convergence.
steps_per_epoch = ceil(N_train / batch_size)

### 30 Dec 2025
Obtain new pipeline data from the company I interned at.


## 10 Jan 2026
Organised the repository into `raw/`, `src/`, `hpo/`, `runs/`, and supporting folders.
Reviewed the available industrial files, including CSV time series and engineering spreadsheets.
Confirmed that forecasting would focus on pressure-related variables rather than attempting a full mechanistic pipeline simulation.

## 22 Jan 2026- Raw data inspection
Inspected uploaded spreadsheet files and verified that they contain multiple worksheets and engineering formulas.
Confirmed that the spreadsheets are suitable as engineering references but not ideal as direct modelling inputs.
Decided to build a single consolidated CSV for all forecasting models.

## 23 Jan 2026 - Data processing pipeline
Developed the raw-data processing logic to read industrial CSVs and the point mapping table.
Mapped station tags to engineered features such as inlet/outlet pressure, temperature, fuel-gas consumption, and compressor states.
Constructed a processed hourly modelling dataset and exported it as `raw/数据处理结果.csv`.

## 28 Jan 2026 - Daily dataset construction
Developed a separate `build_day_dataset.py` utility for daily-level aggregation.
Inspected Excel inputs and validated sheet names and structure.
Confirmed that Python can read the uploaded `.xlsx` files once the required packages are installed.

## 6 Feb 2026 - Ridge Regression baseline
Implemented `model1_ridge_train.py`.
Converted time series into flattened supervised windows.
Trained 1-hour and 10-hour forecasting baselines.
Saved metrics and forecast curves under `runs/model1_ridge/`.
Established Ridge Regression as the simplest linear benchmark.

## 10 Feb 2026 - CNN1D implementation
Implemented `model2_cnn1d_train.py`.
Added support for:
configurable lookback and horizon,
delta or absolute target,
optional target-lag feature,
optional removal of compressor-state features,
standard artifact output (`.pt`, `.json`, `.png`, `.csv`).
Added `model2_cnn1d_hpo.py` for Optuna-based tuning.
Tested both absolute-target and delta-target versions.

## 20 Feb 2026 - CNN1D key finding
Observed that direct absolute-target forecasting produced poor generalisation on strict chronological splits.
Introduced `target_mode=delta`, predicting pressure change instead of absolute pressure.
Found that delta learning significantly improved short-horizon test performance.
Performed ablations:
with and without target lag,
horizon = 1 and horizon = 10.
Concluded that delta mode should be retained for neural models.

## 25 Feb 2026 - LSTM model development
Implemented `model4_lstm_train.py`.
Added recurrent forecasting with configurable hidden size, layer count, dropout, learning rate, and batch size.
Added `model4_lstm_hpo.py` for Optuna-based search.
Run HPO and final training using the best parameter set.
Generated checkpoint, report JSON, forecast curve, loss curve, and test prediction CSV.

## 27 Feb 2026 - BiLSTM baseline
Implemented `model3_bilstm_train.py`.
Added bidirectional recurrent sequence modelling.
Created `model3_bilstm_hpo.py`.
Run baseline and HPO-tuned BiLSTM in delta mode with target lag.
Achieved short-horizon test performance comparable to CNN1D and LSTM.

## 4 Mar 2026 - BiLSTM refinement attempt (negative result)
Modified the BiLSTM code to:
use absolute-space validation RMSE for early stopping,
replace MSE with Huber loss (SmoothL1Loss),
add weight decay,
log both `val_rmse_abs` and `val_rmse_scaled`.
Re-ran HPO with the updated objective.
Result:
validation RMSE improved strongly,
but test RMSE became worse than the earlier BiLSTM run.
Interpretation:
the refined BiLSTM overfitted the validation segment and generalised less well to the held-out test period.
Decision:
keep the earlier BiLSTM configuration as the final BiLSTM result,
document the refined variant explicitly as an unsuccessful but informative experiment.

## 6 Mar 2026 - CNN-BiLSTM model
Implemented `model5_cnn_bilstm_train.py`.
Combined temporal convolutions with a bidirectional recurrent stack.
Implemented `model5_cnn_bilstm_hpo.py`.
Fixed Optuna search-space issues (e.g. valid log-scale bounds for weight decay).
Run HPO and final training.
Confirmed that the model performs competitively, with short-horizon RMSE in the same range as other delta-based neural models.

## Entry 13 - Metrics interpretation and reporting logic
Clarified the distinction between scaled-space metrics and original-scale metrics.
Confirmed that the final reported MAE/RMSE/R^2 should be interpreted in the original physical scale after inverse transformation and delta reconstruction.
Prepared textual justification for using MAE, RMSE, and R^2 together:
MAE for typical error,
RMSE for sensitivity to large deviations,
R^2 for overall fit relative to a baseline.

## 10 Mar 2026
Drafted a detailed project report.
Rewrote the repository README to match the final structure and methodology.

## 11 Mar 2026
All major model families are implemented:
  - Ridge Regression
  - CNN1D
  - LSTM
  - BiLSTM
  - CNN-BiLSTM
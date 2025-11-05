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


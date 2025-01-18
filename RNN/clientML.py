import os
import time
import re
from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from math import sqrt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import logger
from common import containers_info


dataset = os.environ.get("DATASET")
logging_level = logger.getLevelName(os.environ.get("LOGGING_LEVEL"))

x_train = []
y_train = []
x_test = []
y_test = []
scaler = None
global_filename = ""
global_date = ""
occupied = 0


# Configure logging
log = logger.setup_logger(containers_info.get_current_container_name(), logging_level)


# # [] -> actual parameters, 0 -> value of the metric
# parameters_best = ([],0)
# parameters_before_last_agg = ([], 0)

# Create a custom dataset class for PyTorch DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out



def get_data():
    ##########################################################################################
    # ASK TO ANOTHER ENTITY WICH IS THE DATASET FILE AND OTHER RESTRICTIONS
    file_name = dataset
    date = "2024-05-16"
    ##########################################################################################

    global global_filename
    global_filename = str(file_name).split(".")[0]
    global global_date
    global_date = str(date)

    df = pd.read_csv("/app/dataset/" + str(file_name), index_col="timestamp", parse_dates=True)
    log.info("df")
    log.info(df)
    day = pd.to_datetime(str(date))
    #df = df[day - timedelta(days=28) :]

    # Preprocessing
    global scaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Create sequences and labels for training
    global seq_length
    seq_length = int(0.5 * len(df))
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i + seq_length])
        y.append(df_scaled[i + seq_length])


    X, y = np.array(X), np.array(y)

    
    global train_size
    train_size = int(0.8 * len(X))
    global x_train
    global y_train
    global x_test
    global y_test
    x_train, x_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    
    log.info("Get data successfully!")

    return len(x_train)

##############################################################################
##############################################################################
##############################################################################
def fit(config):
    log.info("----------------  FIT  ----------------- ")
    log.debug("config: " + str(config))
    global scaler
    global y_test
    global x_test
    global x_train
    log.info(type(x_train))
    log.info(x_train.shape)
    input_size = x_train.shape[2]
    hidden_size = 128
    output_size = 1
    learning_rate = 0.001
    num_epochs=int(config["epochs"]),
    batch_size=int(config["batch_size"]),
    validation_data=(x_test, y_test),
    
    # Create data loaders
    train_dataset = TimeSeriesDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size[0]), shuffle=True)
    test_dataset = TimeSeriesDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size[0]), shuffle=False)


    # Initialize the model, loss function, and optimizer
    global modelRNN
    modelRNN = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelRNN.parameters(), lr=learning_rate)
    log.info(modelRNN)
    # Lists to store loss values
    train_losses = []
    test_losses = []
   # Training the model
    for epoch in range(num_epochs[0]):
        modelRNN.train()  # Set the model to training mode
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            outputs = modelRNN(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        # Average training loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Evaluate on test data
        modelRNN.eval()  # Set the model to evaluation mode
        running_test_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = modelRNN(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)

        # Average test loss for this epoch
        avg_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs[0]}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    # Plot the training and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs[0] + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs[0] + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return len(x_test), {"input_size": input_size, "hidden_size": hidden_size, "input_size": output_size, "learning_rate":learning_rate,
                        "num_epochs":num_epochs, "batch_size": batch_size}

def evaluate(config):
    log.info("----------------  EVALUATE  ----------------- ")
    log.debug("config: " + str(config))

    global modelRNN
    global x_test
    global y_test

    start_time_evaluate = time.time()
    loss, r_square = modelRNN.evaluate(x_test, y_test, steps=int(config["val_steps"]))
    log.info("Evaluate Duration: " + str((time.time() - start_time_evaluate)))

    return float(loss), len(x_test), {"r2_score": float(r_square)}

def predict(plot_graphs, before_after):
    global modelRNN
    global y_test

    # Create predictions
    with torch.no_grad():
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_pred = modelRNN(X_test_tensor).numpy()
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test)
    # Inverse transform y_test after prediction calculations
   
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)

    # Create today's path based on the current date
    today = datetime.now()
    today_path = f"/app/logs/{today}"
    if not os.path.exists(today_path):
        os.makedirs(today_path)

    # Define the log file path
    log_file = os.path.join(today_path, 'metrics_log.txt')

    # Write metrics to the file
    with open(log_file, 'a') as f:
        f.write(f"Dataset: {global_filename}")
        f.write(f"Evaluation MSE: {mse:.3f}\n")
        f.write(f"Evaluation RMSE: {rmse:.3f}\n")
        f.write(f"Evaluation MAE: {mae:.3f}\n")
        f.write(f"Evaluation R2_SCORE: {r2:.3f}\n")
        f.write(f"Evaluation Explained Variance: {explained_variance:.3f}\n")
        f.write("\n")  # Add a newline for separation between log entries

     # Extract the timestamps for the test set
    df = pd.read_csv(f"/app/dataset/{dataset}", index_col="timestamp", parse_dates=True)
    test_timestamps = df.index[train_size + seq_length:]

     # Save plot and predictions
    today_path = RNNplot_graphs_fn(str(global_filename), str(global_date), y_test, y_pred, today, today_path, test_timestamps)
    
    today = datetime.now()
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv(f"{today_path}/{today}.csv", index=False)

    # Return the results
    return {
        "mse": float(mse), 
        "mae": float(mae),
        "explained_variance": float(explained_variance)
    }, {
        "r2": float(r2)
    }

def get_prediction_for_day_hour(matrix, day, hour):
    """
    Get the prediction for a specific day and hour from the weekly predictions matrix.
    
    Parameters:
    - matrix (numpy array): The result matrix of shape (168, 1) with weekly predictions.
    - day (int): Day of the week (0=Monday, 6=Sunday).
    - hour (int): Hour of the day (0 to 23).
    
    Returns:
    - float: The prediction for the specified day and hour.
    """
    # Calculate the index
    index = day * 24 + hour
    
    # Ensure the index is within the valid range
    if index < 0 or index >= matrix.shape[0]:
        raise ValueError("Invalid day or hour value.")
    
    # Access the prediction
    return matrix[index, 0]

def predictOne(day, hour):
    with torch.no_grad():
        #log.info(type(x_test))
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        #log.info(type(X_test_tensor))
        #log.info(X_test_tensor)
        y_pred = modelRNN(X_test_tensor).numpy()
        #log.info(type(y_pred))
        #log.info(type(scaler))
        #log.info(y_pred)
        y_pred = scaler.inverse_transform(y_pred)

    
    y_pred_df = pd.DataFrame(y_pred)
    prediction = get_prediction_for_day_hour(y_pred_df,day,hour)
    
    log.info(prediction)
    return {"prediction": str(prediction)}

def RNNplot_graphs_fn(file_name, global_date, y_test, y_pred, today, today_path, timestamps):
    plt.figure(figsize=(20, 10))
    
    # Ensure timestamps and predictions match in length
    assert len(timestamps) == len(y_test), "Timestamps and y_test length must match!"
    
    # Plot actual and predicted values with timestamps
    plt.plot(timestamps, y_test, label='Actual', color='blue')
    plt.plot(timestamps, y_pred, label='Predicted', color='orange')
    
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Recurrent Neural Networks Prediction")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Save plot
    pattern = re.compile(r'^run(\d+)\.json$')
    count = max(
        [int(pattern.match(path).group(1)) for path in os.listdir(today_path) if pattern.match(path)], 
        default=-1
    ) + 1
    
    runi = f"run{count}"
    plt.savefig(f"{today_path}/predict_{file_name}_{runi}.png")
    
    return today_path

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import explained_variance_score
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


def get_parameters():
    log.info("Getting Parameters!")
    global model
    parameters = model.get_weights()

    return parameters



def set_parameters(parameters):
    global model
    model.set_weights(parameters)
    log.info("Parameters Set!")


def in_usage(value):
    global occupied
    occupied = value
    if value == 1:
        log.info("ML app is now in use, and blocked")
    elif value == 0:
        log.info("ML app is now freed")

def get_data():
       ##########################################################################################
    # ASK TO ANOTHER ENTITY WICH IS THE DATASET FILE AND OTHER RESTRICTIONS
    file_name = dataset
    date = "2022-05-16"
    ##########################################################################################

    global global_filename
    global_filename = str(file_name).split(".")[0]
    global global_date
    global_date = str(date)

    df = pd.read_csv("/app/dataset/" + str(file_name), index_col="timestamp", parse_dates=True)
    day = pd.to_datetime(str(date))
    #df = df[day - timedelta(days=28) :]
    # Calculate size for splitting the DataFrame
    size = int(len(df) * 0.8)
    
    global SARIMA_train
    global SARIMA_test
    # Split into train and test sets
    SARIMA_train, SARIMA_test = df[0:size], df[size:len(df)]
    SARIMA_train.index = pd.DatetimeIndex(SARIMA_train.index).to_period('D')


    log.info("Get data successfully!")

    return len(SARIMA_train)

##############################################################################
##############################################################################
##############################################################################
def evaluate(config):
    log.info("----------------  EVALUATE  ----------------- ")
    log.debug("config: " + str(config))

    global modelA
    global x_test
    global y_test

    start_time_evaluate = time.time()
    loss, r_square = modelA.evaluate(x_test, y_test, steps=int(config["val_steps"]))
    log.info("Evaluate Duration: " + str((time.time() - start_time_evaluate)))

    return float(loss), len(x_test), {"r2_score": float(r_square)}

def fit(config):
    global SARIMA_test
    global SARIMA_train
    
    log.info("----------------  FIT  ----------------- ")
    log.debug("config: " + str(config))
    
    single_column_df = SARIMA_train[['sum']]
    series = single_column_df.squeeze()

    # Ensure no NaN values are present in the training data
    if series.isnull().any():
        raise ValueError("SARIMA_train contains NaN values")

    testvalues = SARIMA_test.values.tolist()
    global modelS
    
    # Initialize and fit the SARIMA model with the initial history
    # Assuming seasonal_order (P, D, Q, m) is specified in config
    seasonal_order = config.get('seasonal_order', (2, 1, 2, 7))  # Example default seasonal order
    order = config.get('order', (2, 1, 2))  # Example default order
    
    modelS = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    modelS = modelS.fit(method='nm', maxiter=2000)

    results = {
        "loss": 0,
        "r2_score": 0,
        "val_loss": 0,
        "val_r2_score": 0,
    }
    return len(SARIMA_train), results

def predict(plot_graphs, before_after):
   from sklearn.metrics import explained_variance_score

def predict(plot_graphs, before_after):
    global modelS
    global SARIMA_test
    global SARIMA_train

    predictions = list()
    
    testvalues = SARIMA_test.values.tolist()
    testtimestamps = SARIMA_test.index.tolist()
    SARIMA_train = SARIMA_train.values.tolist()
    log.info(modelS.model_orders['ar'])
    
    for t in range(len(SARIMA_test)):
        modelS = SARIMAX(SARIMA_train, order=(modelS.model_orders['ar'],1,2), 
                         seasonal_order=(1,1,1,7))
        modelS_fit = modelS.fit(method='nm', maxiter=2000)
        output = modelS_fit.forecast()
        
        yhat = output[0]
        predictions.append(yhat)
        
        SARIMA_train.append(testvalues[t])
    
    predictions_df = pd.DataFrame({'Timestamp': testtimestamps, 'Predictions': predictions})
    
    # Convert the 'Timestamp' column to datetime
    predictions_df['Timestamp'] = pd.to_datetime(predictions_df['Timestamp'])
    
    # Set the 'Timestamp' column as the index
    predictions_df.set_index('Timestamp', inplace=True)
    
    # Calculate metrics
    mse = mean_squared_error(SARIMA_test, predictions)
    rmse = sqrt(mse)
    mae = mean_absolute_error(SARIMA_test, predictions)
    r2 = r2_score(SARIMA_test, predictions)
    explained_variance = explained_variance_score(SARIMA_test, predictions)
    # Save Logs to a file
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
        f.write(f"Evaluation R2SCORE: {r2:.3f}\n")
        f.write(f"Evaluation Explained Variance: {explained_variance:.3f}\n")
        f.write("\n")  # Add a newline for separation between log entries
    # Plot the results
    SARIMAplot_graphs_fn(str(global_filename), str(global_date), SARIMA_test, predictions_df, today_path, today)

    results = {
        "loss": rmse,
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "explained_variance": explained_variance,
        "val_loss": 0,
        "val_r2_score": 0,
    }

    return len(SARIMA_test), results


def evaluate(config):
    log.info("----------------  EVALUATE  ----------------- ")
    log.debug("config: " + str(config))

    global x_test
    global y_test

    start_time_evaluate = time.time()
    loss, r_square = model.evaluate(x_test, y_test, steps=int(config["val_steps"]))
    log.info("Evaluate Duration: " + str((time.time() - start_time_evaluate)))

    return float(loss), len(x_test), {"r2_score": float(r_square)}


def SARIMAplot_graphs_fn(file_name, global_date, ARIMA_history, predictions, today_path, today):
    log.info(str(ARIMA_history))
    log.info(str(predictions))
    plt.figure(figsize=(20, 10))
    plt.plot(ARIMA_history, label="real")
    plt.plot(predictions, label="prediction")
    plt.legend()
    plt.title("SARIMA predictions ")
    
    
    predictions.to_csv(f"{today_path}/{today}.csv")

    # Regular expression to match files like "run0.json", "run1.json", etc.
    pattern = re.compile(r'^run(\d+)\.json$')

    # Find all files that match the pattern and extract the number part
    count = -1
    for path in os.listdir(today_path):
        if os.path.isfile(os.path.join(today_path, path)):
            if pattern.match(path):
                count += 1

    count = 0 if count == -1 else count

    runi = f"run{count}"

    plt.savefig(today_path + "/predict_" + file_name + "_" + runi + ".png")

def predictOne(day, hour):
    global modelA
    global ARIMA_test
    global ARIMA_train

    predictions = list()
    # walk-forward validation
    log.info("This is length of arima_test "+ str(len(ARIMA_test)))
    log.info("This is type of arima_test "+ str(type(ARIMA_test)))
    log.info("This is type of arima_history "+ str(type(ARIMA_train)))
    testvalues = ARIMA_test.values.tolist()

    modelA = SARIMAX(ARIMA_train, order=(5,1,0))
    modelA_fit = modelA.fit(method='nm', maxiter=2000)
    output = modelA_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
 
    #log.info('predicted=%f, expected=%f' % (yhat, obs))
    results ={
        "loss": 0,
        "r2_score":0,
        "val_loss":0,
        "val_r2_score": 0,
    }
    return len(ARIMA_test), results


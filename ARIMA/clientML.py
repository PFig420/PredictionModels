import os
import time
import re
from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf


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



def get_data():
    ##########################################################################################
    # ASK TO ANOTHER ENTITY WHICH IS THE DATASET FILE AND OTHER RESTRICTIONS
    file_name = dataset
    date = "2022-05-10"
    ##########################################################################################

    global global_filename
    global_filename = str(file_name).split(".")[0]
    global global_date
    global_date = str(date)

    df = pd.read_csv("/app/dataset/" + str(file_name), index_col="timestamp", parse_dates=True)
    day = pd.to_datetime(str(date))
    
    # Calculate size for splitting the DataFrame
    size = int(len(df) * 0.8)

    # Check for stationarity (Dickey-Fuller test)
    result = adfuller(df[['sum']])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # If non-stationary, perform differencing
    if result[1] > 0.05:  # Assuming a significance level of 0.05
        df['diff'] = df['sum'].diff().dropna()
    else:
        df['diff'] = df['sum']
    
    # Apply noise injection to the training data
    df['diff_with_noise'] = add_noise(df['diff'], noise_level=0.01)

    global ARIMA_train
    global ARIMA_test
    # Split into train and test sets
    ARIMA_train, ARIMA_test = df[0:size], df[size:len(df)]

    log.info("Get data successfully!")

    return len(ARIMA_train)



def fit(config):
    global ARIMA_test
    global ARIMA_train
    global series
    global p
    global q

    log.info("----------------  FIT  ----------------- ")
    log.debug("config: " + str(config))

    # Use noisy data for ARIMA training
    single_column_df = ARIMA_train[['diff']]
    series = single_column_df.squeeze()

    # Calculate ACF and PACF values
    acf_vals = acf(series.dropna(), nlags=5)
    pacf_vals = pacf(series.dropna(), nlags=5)
    log.info(f"len of series: {len(series)}")
    log.info(f"len of acf: {len(acf_vals)}")
    log.info(f"len of pacf: {len(pacf_vals)}")
    #p = np.where(np.abs(pacf_vals) < 1.96 / np.sqrt(len(series.dropna())))[0][0]
    p = 1
    q = 1
    #q = np.where(np.abs(acf_vals) < 1.96 / np.sqrt(len(series.dropna())))[0][0]

    print('Determined p:', p)
    print('Determined q:', q)

    # Initialize and fit the ARIMA model with the noisy training data
    global modelA
    modelA = ARIMA(series, order=(p, 1, q))
    modelA = modelA.fit()

    results = {
        "loss": 0,
        "r2_score": 0,
        "val_loss": 0,
        "val_r2_score": 0,
    }
    return len(ARIMA_train), results


def predict(plot_graphs, before_after):
    global modelA
    global ARIMA_test
    global ARIMA_train
    global series
    global p
    global q

    predictions = []
    # walk-forward validation
    testvalues = ARIMA_test.values.tolist()
    testtimestamps = ARIMA_test.index.tolist()

    for t in range(len(ARIMA_test)):
        print(series.dtypes)
        modelA = ARIMA(series, order=(p, 1, q))
        modelA_fit = modelA.fit()
        output = modelA_fit.forecast()
        
        
        if output.size == 0:
            continue
        else:
            yhat = output.iloc[0]
        predictions.append(yhat)
        series.loc[len(series)] = testvalues[t][0]

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({'Timestamp': testtimestamps, 'Predictions': predictions})
    predictions_df['Timestamp'] = pd.to_datetime(predictions_df['Timestamp'])
    predictions_df.set_index('Timestamp', inplace=True)
    log.info(predictions)
    log.info(predictions_df)
    log.info(ARIMA_test)
    # Calculate evaluation metrics
    mse = mean_squared_error(ARIMA_test["sum"], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ARIMA_test["sum"], predictions)
    r2 = r2_score(ARIMA_test["sum"], predictions)
    explained_variance = explained_variance_score(ARIMA_test["sum"], predictions)

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
        f.write(f"Evaluation R2SCORE: {r2:.3f}\n")
        f.write(f"Evaluation Explained Variance: {explained_variance:.3f}\n")
        f.write("\n")  # Add a newline for separation between log entries


    # Plot predictions vs actual values
    ARIMAplot_graphs_fn(str(global_filename), str(global_date), ARIMA_test, predictions_df, today_path, today)


    results ={
        "loss": 0,
        "r2_score":0,
        "val_loss":0,
        "val_r2_score": 0,
    }

    return len(ARIMA_test), results


def evaluate(config):
    log.info("----------------  EVALUATE  ----------------- ")
    log.debug("config: " + str(config))

    global x_test
    global y_test

    start_time_evaluate = time.time()
    loss, r_square = model.evaluate(x_test, y_test, steps=int(config["val_steps"]))
    log.info("Evaluate Duration: " + str((time.time() - start_time_evaluate)))

    return float(loss), len(x_test), {"r2_score": float(r_square)}


def ARIMAplot_graphs_fn(file_name, global_date, ARIMA_history, predictions, today_path, today):
    log.info(str(ARIMA_history))
    log.info(str(predictions))
    plt.figure(figsize=(20, 10))
    plt.plot(ARIMA_history['diff'], label="real")
    plt.plot(predictions, label="prediction")
    plt.legend()
    plt.title("ARIMA prediction")
    
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

    modelA = ARIMA(ARIMA_train, order=(5,1,0))
    modelA_fit = modelA.fit()
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


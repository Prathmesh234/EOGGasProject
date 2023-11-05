import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from functools import reduce

def process(filename="sensor_readings.csv"):
    readings = pd.read_csv(filename)
    
    for col in readings.columns:
        if col == "time" or col == "Unnamed: 0":
            continue
        mean_value = readings[col].mean()
        std_dev = readings[col].std()
        
        readings[col] = np.abs((readings[col] - mean_value) / std_dev)

    return readings

def get_anomaly_data(readings):
    threshold = 2.5
    anomaly_indices = set()
    for col in readings.columns:
        if col == "time" or col == "Unnamed: 0":
            continue
        filtered_indices = set(readings.loc[readings[col] > threshold].index)
        anomaly_indices = anomaly_indices.union(filtered_indices)
    
    return readings.loc[list(anomaly_indices)]

def populate_Y_data(readings):
    locations = pd.read_csv("leak_locations_and_rate.csv")
    df = locations[locations["EmissionCategory"] == "Fugitive"]
    
    class_intervals = {}
    indices = []
    for i, row in df.iterrows():
        id = row["LeakPointId"][:2]
        if id not in class_intervals.keys():
            class_intervals[id] = [[row["tStart"], row["tEnd"]]]
        else:
            class_intervals[id] += [[row["tStart"], row["tEnd"]]]
    for id, intervals in class_intervals.items():
        readings["Y_" + id] = np.zeros(len(readings))
        for v in intervals:
            readings.loc[(readings["time"] >= v[0]) & (readings["time"] <= v[1]), "Y_" + id] = 1

def get_weather_data():
    weather_data = pd.read_csv("weather_data.csv")
    weather_data.drop(columns=["timestamp"], inplace = True)
    # weather_data = weather_data[:len(readings)]
    return weather_data

def get_X_data(readings):
    X_features = []
    for col in readings.columns:
        if col == "time" or col == "Unnamed: 0":
            continue
        X_features += [col]
        
    return readings[X_features]
    # return pd.concat([readings[X_features], weather_data], axis =  1)

def train_EOG(readings):
    X_temp = get_X_data(readings)
    weather_data = get_weather_data()
    weather_data = weather_data[:len(X_temp)]
    X = pd.concat([X_temp, weather_data], axis =  1)
    populate_Y_data(readings)
    classes = ["4S", "4T", "4W", "5S", "5W"]
    Ys = [readings["Y_" + id] for id in classes]
    Y_preds = []
    models = []
    for y, id in zip(Ys, classes):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        model.fit(X_train, y_train)
        models += [model]
        
        y_pred = model.predict(X_test)
        Y_preds += [y_pred]
        accuracy = accuracy_score(y_test, y_pred)
        # print(id, accuracy)
    
    # print("Model accuracy: ", (Ys == Y_preds).result = reduce(lambda x, y: x & y, vector) / len(Ys))
    # conf_matrix = confusion_matrix(Ys, Y_preds)
    # print("Confusion Matrix:", conf_matrix)
    return models

def train_historic():
    readings = process("sensor_readings.csv")
    return train_EOG(readings)

models = train_historic()
test_readings = process("sensor_readings.csv") # To be replaced with the input file path

result = {}
result["time"] = test_readings["time"]
result_df = pd.DataFrame(result)
test_readings.drop(columns = ["Unnamed: 0", "time"], inplace=True)
weather_data = get_weather_data()
weather_data = weather_data[:len(test_readings)]
X_test = pd.concat([test_readings, weather_data], axis =  1)
classes = ["4S", "4T", "4W", "5S", "5W"]
Y_pred = []
for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    Y_pred += [y_pred]

Y_pred_df = pd.DataFrame(Y_pred).T
answer = []
for i,row in Y_pred_df.iterrows():
    res = ""
    for i,y in enumerate(row):
        if y == 1:
            if not res:
                res += classes[i]
            else:
                res += "|" + classes[i]
    if not res:
        res = "None"
    answer += [res]
result_df["location"] = answer
result_df.to_csv("results.csv")

import numpy as np
import pandas as pd
import tensorflow as tf

"""
Goal of this project is to create a neural network that uses stock data as inputs and give an output based on if the stock will increase or decrease in value the next day. 
Using x stock returns throughout the years, this will be the initial dataset I use to experiment and create the neural network structure using.

Stock data used only ever goes up to April 4th, 2020.
"""

AAPL = pd.read_csv("AAPL.csv")
AMC = pd.read_csv("AMC.csv")

def output_data(input_data):
    np1 = input_data["Open"].shift(-1)
    n = input_data["Close"]
    df = pd.DataFrame(columns = ["Signal"])

    r = np1 - n
    df["Signal"] = r/abs(r)
    df["Signal"] = df["Signal"].replace(-1, 0)
    df["Signal"] = df["Signal"].astype(bool)
    return df["Signal"]

def input_data(raw_data, ndays = 5):
    # x = raw_data.copy()
    x = pd.DataFrame(columns = ["Dev", "Mean Value"])
    x["Dev"] = raw_data["High"] - raw_data["Low"]
    x["Std of Dev"] = x["Dev"].rolling(ndays).std()
    x["Mean Dev"] = x["Dev"].rolling(ndays).mean()
    x["Std"] = raw_data ["Close"].rolling(ndays).std()
    x["Std of Std"] = x["Std"].rolling(ndays).std()
    return x

def data_gen(raw_data, ndays = 5, split_ratio = .2):
    x = input_data(raw_data, ndays)[ndays*2:-2]
    y = output_data(raw_data)[ndays*2:-2]

    split_at = round(len(x)*(1-split_ratio))
    xtest, ytest = x[split_at:], y[split_at:]
    xtrain, ytrain = x[:split_at], y[:split_at]

    return(xtrain, xtest, ytrain, ytest)


#Creating the neural network model

xtrain, xtest, ytrain, ytest = data_gen(AAPL)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation = "relu", input_dim = len(xtrain.columns)),
    tf.keras.layers.Dense(256, activation = "tanh"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(
    optimizer = "sgd",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

model.fit(
    x = xtrain, y = ytrain, epochs = 20
)

# Testing for generalization

def generalization_eval(dataframe):

    x = input_data(dataframe)
    y = output_data(dataframe)

    y_hat = model.predict(x)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    correct = sum(y == y_hat)
    total = len(y)
    per_correct = correct/total
    out = output_ratio(y_hat)

    return(per_correct*100, out)

def output_ratio(y_hat):
    pos = sum(y_hat)
    neg = len(y_hat) - pos
    if neg == 0:
        return np.inf
    else:
        return pos/neg

correct, outrat = generalization_eval(AMC)
print(correct, outrat)

# Model has optimized to exclusively predict 1, not accurately reflecting an intuitive approach.

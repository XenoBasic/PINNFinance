import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

"""
Goal of this project is to create a neural network that uses stock data as inputs and give an output based on if the stock will increase or decrease in value the next day. 
Using x stock returns throughout the years, this will be the initial dataset I use to experiment and create the neural network structure using.

Stock data used only ever goes up to April 4th, 2020.
"""

# Data Initiation

AAPL = pd.read_csv("AAPL.csv")
AMC = pd.read_csv("AMC.csv")

def output_data(input_data):
    np1 = input_data["Close"].shift(-1).dropna() # next day opening price
    return np1

def input_data(df):
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()[:len(df)-1]

def data_gen(raw_data, split_ratio = .1):
    x = input_data(raw_data)
    y = output_data(raw_data)

    split_at = round(len(x)*(1-split_ratio))
    xtest, ytest = x[split_at:], y[split_at:]
    xtrain, ytrain = x[:split_at], y[:split_at]

    return(xtrain, xtest, ytrain, ytest)


#Creating the neural network model
def base_model():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation = "linear"),
        tf.keras.layers.Dense(256, activation = "elu"),
        tf.keras.layers.Dense(1)])

    model.compile(
        optimizer = "adam",
        loss = "MSE",
        metrics = ["MSE"]
    )   
    return model

@tf.keras.utils.register_keras_serializable("pinn_loss")
def PINN_loss(y_true, y_pred):
    mean = tf.reduce_mean(y_true)
    stdev = tf.math.reduce_std(y_true)
    MSE = tf.square(y_true - y_pred)
    PDF = (1/tf.sqrt(2*np.pi*tf.square(stdev)))*tf.exp(-tf.square(y_pred-mean)/(2*tf.square(stdev)))
    return (MSE + (MSE * (1-PDF)))

def PINN_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation = "linear"),
        tf.keras.layers.Dense(256, activation = "elu"),
        tf.keras.layers.Dense(1)])

    model.compile(
        optimizer = "adam",
        loss = PINN_loss,
        metrics = ["MSE"]
    )   
    return model

def model_training(data, model = base_model()):
    checkpoint_path = "weights/base.keras"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                     save_best_only=True,
                                                     monitor = "loss",
                                                     verbose = 1)
    
    xtrain, xtest, ytrain, ytest = data_gen(data)

    model.fit(xtrain,
              ytrain,
              epochs = 500,
              callbacks = [cp_callback]
            )
    
    return model

# Testing Area
# model = model_training(AAPL)

model = tf.keras.models.load_model("weights/PINN.keras")
AAPL_raw = pd.read_csv("AMC.csv")
AAPL_input = input_data(AAPL_raw)
AAPL_output = output_data(AAPL_raw)
AAPL_predictions = model.predict(AAPL_input)
model.evaluate(AAPL_input, AAPL_output)
plt.plot(AAPL_predictions, color = "r")
plt.plot(AAPL_output, color = "y")
plt.legend()
plt.show()

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
#Creating a custom passthrough variable layer class
@tf.keras.utils.register_keras_serializable(package="pinn_loss")
class PassthroughFeature(tf.keras.layers.Layer):
    def __init__(self, feature_index=3, **kwargs):
        """
        Docstring for __init__
        
        :param self: Description
        :param feature_index: Description
        :param kwargs: Description
        """
        super().__init__(**kwargs)
        self.feature_index = feature_index

    def call(self, inputs):
        """
        Docstring for call
        
        :param self: Description
        :param inputs: Description
        """
        return inputs[:, self.feature_index:self.feature_index+1]
    
    def get_config(self):
        """
        Docstring for get_config
        
        :param self: Description
        """
        config = super().get_config()
        config.update({"feature_index": self.feature_index})
        return config

def base_model():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation = "silu"),
        tf.keras.layers.Dense(256, activation = "gelu"),
        tf.keras.layers.Dense(1)])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss = "MSE",
        metrics = ["MSE"]
    )   
    return model

@tf.keras.utils.register_keras_serializable("pinn_loss")
def PINN_loss(y_true, y_pred):
    target_price = y_true[:, 0:1]
    predicted_price = y_pred[:, 0:1]
    prev_close = y_pred[:, 1:2]

    MSE = tf.reduce_mean(tf.square(target_price - predicted_price))

    std = tf.math.reduce_std(prev_close)
    mean = tf.reduce_mean(prev_close)
    z_score = 2.576
    t = 1/5
    mini = tf.exp((mean - 0.5*tf.square(std))*t-z_score*std*np.sqrt(t))
    maxi = tf.exp((mean - 0.5*tf.square(std))*t+z_score*std*np.sqrt(t))
    pred_min = prev_close * mini
    pred_max = prev_close * maxi

    lower_pen = tf.nn.relu(pred_min - y_pred)
    upper_pen = tf.nn.relu(y_pred - pred_max)

    lower_pen = tf.square(lower_pen)
    upper_pen = tf.square(upper_pen)
    range_penalty = tf.reduce_mean(lower_pen + upper_pen)

    negative_penalty = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))
    
    return MSE + 100*range_penalty + 10000 * negative_penalty

#a new metric for falling outside of a range
@tf.keras.utils.register_keras_serializable("range_penalty")
def range_penalty_metric(y_true, y_pred):
    predicted_price = y_pred[:, 0:1]
    prev_close = y_pred[:, 1:2]
    
    std = tf.math.reduce_std(predicted_price) + 1e-5
    mean = tf.reduce_mean(predicted_price)
    t = 1/5
    z_score = 2.576

    exp_lower = tf.clip_by_value((mean - 0.5*tf.square(std))*t - z_score*std*tf.sqrt(t), -5.0, 5.0)
    exp_upper = tf.clip_by_value((mean - 0.5*tf.square(std))*t + z_score*std*tf.sqrt(t), -5.0, 5.0)
    
    pred_min = prev_close * tf.exp(exp_lower)
    pred_max = prev_close * tf.exp(exp_upper)
    
    lower_pen = tf.nn.relu(pred_min - predicted_price)
    upper_pen = tf.nn.relu(predicted_price - pred_max)
    
    return tf.reduce_mean(tf.square(lower_pen) + tf.square(upper_pen))

@tf.keras.utils.register_keras_serializable("negative_penalty")
def negative_penalty_metric(y_true, y_pred):
    predicted_price = y_pred[:, 0:1]
    return tf.reduce_mean(tf.square(tf.nn.relu(-predicted_price)))

def output_data(input_data):
    np1 = input_data["Close"].shift(-1).dropna() # next day close price
    return np1

# Models will use 5 columns from .csv files from yahoo finance
# if using other files, data will have to be cleaned and modified to fit the 5 column standard
def input_data(df):
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()[:len(df)-1]

# data will be split into testing data and training data at the split ratio
# in other words, split_ratio% of the dataset will be testing data, and the rest will be training data 
def data_gen(raw_data, split_ratio = .1):
    x = input_data(raw_data)
    y = output_data(raw_data)

    split_at = round(len(x)*(1-split_ratio))
    xtest, ytest = x[split_at:], y[split_at:]
    xtrain, ytrain = x[:split_at], y[:split_at]

    return(xtrain, xtest, ytrain, ytest)

# simple function implemented to 
def addFileExt(ticker: str):
    return ticker+".CSV"


def PINN_model(input_shape: int, prev_close_index=3):
    inputs = tf.keras.Input(shape=(input_shape,))
    
    x = inputs
    x = tf.keras.layers.Dense(256, activation="silu")(x)
    x = tf.keras.layers.Dense(256, activation="gelu")(x)
    x = tf.keras.layers.Dense(1)(x)

    prev_close = PassthroughFeature(feature_index=prev_close_index)(inputs)
    outputs = tf.keras.layers.Concatenate()([x, prev_close])

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss = PINN_loss,
        metrics = [PINN_loss]
    )   
    return model

def model_training(data, model = base_model()):
    checkpoint_path = "weights/Base.keras" #change file path before training, otherwise will overwrite files
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                     save_best_only=True,
                                                     monitor = "loss",
                                                     verbose = 2)
    
    xtrain, xtest, ytrain, ytest = data_gen(data)

    model.fit(xtrain,
              ytrain,
              epochs = 200,
              callbacks = [cp_callback]
            )
    
    return model 

# Testing Area
Stock = "AAPL" # stock ticker to train model or test model on
retrain = True # True for training model, False for testing model

stock_data = pd.read_csv(addFileExt(Stock))

if retrain:
    model = model_training(stock_data)

model = tf.keras.models.load_model("weights/Base.keras") #change file path

data_input = input_data(stock_data)
data_output = output_data(stock_data)
value_predictions = model.predict(data_input)
stock_price_predictions = value_predictions[:, 0]  #just the price, ignore passthrough

predicted_returns = pd.DataFrame(stock_price_predictions, columns=["Current Day Close"])
model.evaluate(data_input, data_output)

predicted_returns["Previous Day Close"] = data_input["Close"]
rp = (predicted_returns["Current Day Close"] - predicted_returns["Previous Day Close"]) / predicted_returns["Previous Day Close"]
signalp = (rp/abs(rp))
# print(predicted_returns)

actual_returns = pd.DataFrame(data_output)
actual_returns["Previous Day Close"] = data_input["Close"]
ra = (actual_returns["Close"] - actual_returns["Previous Day Close"]) / actual_returns["Previous Day Close"]
signala = ra/abs(ra)
# print(signala)

print(sum(signala == signalp)/len(signala))

# matplotlib graphing
fig, ax = plt.subplots()
ax.plot(stock_price_predictions, color = "r", label="Predicted Value")
ax.plot(data_output, color = "y", label="Actual Value")
plt.xlabel("Time")
plt.title(Stock)
plt.ylabel("Price")
ax.set_xlim(left=0)
ax.axhline(y=0, color='black', linewidth=2.5, linestyle='-')
ax.axvline(x=0, color='black', linewidth=2.5, linestyle='-')
plt.legend()
plt.grid()
plt.show()
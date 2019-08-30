#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:07:58 2019

@author: jprieto
"""

#import all essential libraries
import sys 
import numpy # linear algebra

import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('fechas.csv')
#df['FECHA'] = pd.to_datetime(df['FECHA'])
nv =  df['BANDERA_INCENDIO']#df.iloc[:,-1]
nv = pd.DataFrame(data=nv)
df = df.set_index('FECHA')

#nv = nv.resample('W').mean()

# LSTM for fires

from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.metrics import binary_accuracy

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=3):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility

numpy.random.seed(7)


# load the dataset
dataframe = nv#read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

import keras_metrics
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy','mae',keras_metrics.recall()])

model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])



# calculate root mean squared error
#b_acc = binary_accuracy(testPredict, testY)

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# plot baseline and predictions
plt.figure(figsize=(15,8))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

down = 2300
up = 2500
plt.figure(figsize=(15,8))
plt.plot(scaler.inverse_transform(dataset[down:up]),label='Dataset')
plt.plot(trainPredictPlot[down:up],label='Train pred')
plt.plot(testPredictPlot[down:up],label='Test pred')
plt.xlabel('Days')
plt.ylabel('Fire flag')
plt.legend()
plt.show()


down = 2350
up = 2400

plt.figure(figsize=(15,8))
plt.plot(scaler.inverse_transform(dataset[down:up]),label='Dataset')
plt.plot(trainPredictPlot[down:up],label='Train pred')
plt.plot(testPredictPlot[down:up],label='Test pred')
plt.xlabel('Days')
plt.xticks(numpy.arange(0, 50, step=2))
plt.ylabel('Fire flag')
plt.legend()
plt.show()
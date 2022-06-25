#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import random
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from config import paths, param, hyper

def slidingWindows(data, sequenceLength):
    x = []
    y = []
    for i in range(len(data) - sequenceLength):
        _idx = i
        _x = data[i:(i + sequenceLength)]
        _y = data[i + sequenceLength]
        x.append([_idx, _x])
        y.append([_idx, _y])
    return x, y

def loadData(datasetPath, sequenceLength, testProportion):
    dataset = np.load(datasetPath, allow_pickle=True)
    date = dataset[0]
    label = dataset[1]
    data = dataset[2]


    idx_data_X, idx_data_Y = slidingWindows(data, sequenceLength)
    random.shuffle(idx_data_X)
    index, dataX, dataY, ndate = [], [], [], []
    for _data_X in idx_data_X:
        index.append([_data_X[0]])
        dataX.append(_data_X[1])
        dataY.append([idx_data_Y[_data_X[0]][1]])
        ndate.append([date[_data_X[0] + sequenceLength]])
        
    testSize = int(len(dataX) * testProportion)
    trainSize = len(dataX) - testSize
    trainX, trainY = np.array(dataX[:trainSize]), np.array(dataY[:trainSize])
    for i in range(len(trainX)):
        scaler = MinMaxScaler()
        scaler.fit(trainX[i])
        trainX[i] = scaler.transform(trainX[i])
        trainY[i] = scaler.transform(trainY[i])
    testX, testY = np.array(dataX[trainSize:]), np.array(dataY[trainSize:])
    for i in range(len(testX)):
        scaler = MinMaxScaler()
        scaler.fit(testX[i])
        testX[i] = scaler.transform(testX[i])
        testY[i] = scaler.transform(testY[i])
    dataX = Variable(torch.Tensor(np.array(dataX)))
    dataY = Variable(torch.Tensor(np.squeeze(np.array(dataY))))
    trainX = Variable(torch.Tensor(trainX))
    trainY = Variable(torch.Tensor(np.squeeze(trainY)))
    testX = Variable(torch.Tensor(testX))
    testY = Variable(torch.Tensor(np.squeeze(testY)))
    return label, index, ndate, dataX, dataY, trainX, trainY, testX, testY


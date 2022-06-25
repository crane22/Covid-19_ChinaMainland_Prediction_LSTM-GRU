#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_loader import loadData
from model import RNN
from config import paths, param, hyper

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
datasetPath = paths['Data']
sequenceLength = hyper['SlideWindowSize']
testProportion = param['TestSetProportion']
label, index, ndate, dataX, dataY, trainX, trainY, testX, testY = loadData(datasetPath, sequenceLength, testProportion)
dataX, dataY = np.array(dataX), np.array(dataY)
dataY_unsqueezed = []
for d in dataY:
    dataY_unsqueezed.append([d])
dataY_unsqueezed = np.array(dataY_unsqueezed)
scalers = []
for i in range(len(dataX)):
    sc = MinMaxScaler()
    sc.fit(dataX[i])
    dataX[i] = sc.transform(dataX[i])
    dataY_unsqueezed[i] = sc.transform(dataY_unsqueezed[i])
    scalers.append(sc)
dataX = Variable(torch.Tensor(dataX))
dataY = Variable(torch.Tensor(np.squeeze(dataY_unsqueezed)))

# Load Model
hiddenSize = hyper['HiddenLayers']
numLayers = hyper['NumLayers']
# inputSize = param['InputDimension']
# outputSize = param['OutputDimension']
inputSize = len(list(dataX[0][0]))
outputSize = len(list(dataX[0][0]))

model = RNN(outputSize, inputSize, hiddenSize, numLayers, sequenceLength).to(device)
modelSavePath = paths['Output_Model']
model.load_state_dict(torch.load(modelSavePath))

# Predict current
model.eval()
dataX = dataX.to(device)
dataY = dataY.to(device)
dataPredict = model(dataX)
lossFunc = torch.nn.MSELoss()
loss = lossFunc(dataPredict, dataY).item()
print(loss)

# alignment with index
# numX, numY, numP = np.array(dataX), np.array(dataY), np.array(dataPredict)
numY, numP = np.array(dataY.to("cpu")), np.array(dataPredict.to("cpu").detach().numpy())
# alignedX = [x for x in range(len(index))]
alignedY = [y for y in range(len(index))]
alignedP = [p for p in range(len(index))]
alignedSC = [sc for sc in range(len(index))]
index = np.squeeze(index)
for idx in range(len(index)):
    alignedY[index[idx]] = numY[idx]
    alignedP[index[idx]] = numP[idx]
    alignedSC[index[idx]] = scalers[idx]
    
alignedY, alignedP = np.array(alignedY), np.array(alignedP)
alignedY_unsqueezed, alignedP_unsqueezed = [], []
for d in alignedY:
    alignedY_unsqueezed.append([d])
alignedY_unsqueezed = np.array(alignedY_unsqueezed)
for d in alignedP:
    alignedP_unsqueezed.append([d])
alignedP_unsqueezed = np.array(alignedP_unsqueezed)

# Plot
plotY, plotP = [], []
for i in range(len(index)):
    plotY.append(alignedSC[i].inverse_transform(alignedY_unsqueezed[i]))
    plotP.append(alignedSC[i].inverse_transform(alignedP_unsqueezed[i]))
plotY = np.squeeze(plotY)
plotP = np.squeeze(plotP)
plt.plot(plotY[:, 0])
plt.plot(plotP[:, 0])
plt.savefig(paths['Output_Figure'])
plt.show()

# Predict future
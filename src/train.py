#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from math import ceil
from tqdm import tqdm
from data_loader import loadData
from model import RNN
from config import paths, param, hyper

def train(device, inputX, inputY, model, criterion, optimizer):
    model.train()
    inputX, inputY = inputX.to(device), inputY.to(device)
    
    # Compute prediction error
    predict = model(inputX)
    loss = criterion(predict, inputY)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(device, tInputX, tInputY, ):
    pass


if __name__ == "__main__":
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load data
    datasetPath = paths['Data']
    sequenceLength = hyper['SlideWindowSize']
    testProportion = param['TestSetProportion']
    label, index, ndate, dataX, dataY, trainX, trainY, testX, testY = loadData(datasetPath, sequenceLength, testProportion)
    
    # Create Model
    hiddenSize = hyper['HiddenLayers']
    numLayers = hyper['NumLayers']
    # inputSize = param['InputDimension']
    inputSize = len(list(dataX[0][0]))
    outputSize = param['OutputDimension']
    model = RNN(outputSize, inputSize, hiddenSize, numLayers).to(device)
    
    # Train
    preloadModelFlag = param['PreloadModelFile']
    learningRate = hyper['LearningRate']
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    numEpochs = hyper['NumEpoch']
    batchSize = hyper['BatchSize']
    
    for epoch in range(numEpochs):
        trainSize = len(list(trainX))
        totalBatches = ceil(trainSize / batchSize)
        progressBar = tqdm(range(totalBatches), desc="CurrLoss: ")
        for batchNum in progressBar:
            if batchNum == totalBatches - 1:
                currLoss = train(device, trainX[batchSize*batchNum:trainSize], 
                                 trainY[batchSize*batchNum:trainSize], model, criterion, optimizer)
            currLoss = train(device, trainX[batchSize*batchNum:batchSize*(batchNum+1)], 
                             trainY[batchSize*batchNum:batchSize*(batchNum+1)], model, criterion, optimizer)
            progressBar.set_description("CurrLoss: " + currLoss)
        print("Epoch: %d, loss: %1.5f" % (epoch, currLoss))
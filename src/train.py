#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from tqdm import tqdm
from data_loader import loadData
from model import RNN
from config import paths, param, hyper

def train(device, inputX, inputY, model, criterion, optimizer):
    model.train()
    inputX, inputY = inputX.to(device), inputY.to(device)
    # print("inputX:", inputX.size())
    # print("inputY:", inputY.size())
    
    # Compute prediction error
    predict = model(inputX)
    # print("predict:", predict.size())
    loss = criterion(predict, inputY)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(device, inputX, inputY, model, criterion):
    model.eval()
    inputX, inputY = inputX.to(device), inputY.to(device)
    predict = model(inputX)
    loss = criterion(predict, inputY)
    return loss.item()
    


if __name__ == "__main__":
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    datasetPath = paths['Data']
    sequenceLength = hyper['SlideWindowSize']
    testProportion = param['TestSetProportion']
    label, index, ndate, dataX, dataY, trainX, trainY, testX, testY = loadData(datasetPath, sequenceLength, testProportion)
    
    # Create Model
    hiddenSize = hyper['HiddenLayers']
    numLayers = hyper['NumLayers']
    # inputSize = param['InputDimension']
    # outputSize = param['OutputDimension']
    inputSize = len(list(dataX[0][0]))
    outputSize = len(list(dataX[0][0]))
    
    model = RNN(outputSize, inputSize, hiddenSize, numLayers, sequenceLength).to(device)
    preloadModelFlag = param['PreloadModelFile']
    modelSavePath = paths['Output_Model']
    if preloadModelFlag == True:
        model.load_state_dict(torch.load(modelSavePath))
    
    # Train
    learningRate = hyper['LearningRate']    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    numEpochs = hyper['NumEpoch']
    # batchSize = hyper['BatchSize']
    
    progressBar = tqdm(range(numEpochs))
    minLoss = 2^63 - 1
    for epoch in progressBar:
        currLoss = train(device, trainX, trainY, model, criterion, optimizer)
        loss = test(device, testX, testY, model, criterion)
        progressBar.set_description("Epoch: %d, currloss: %1.5f, loss: %1.5f" % (epoch, currLoss, loss))
        if loss <= minLoss:
            torch.save(model.state_dict(), modelSavePath)
            loss = minLoss
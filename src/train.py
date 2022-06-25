#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from data_loader import loadData
from model import RNN
from config import paths, param, hyper

def train():
    pass

if __name__ == "__main__":
    numEpochs = hyper['NumEpoch']
    learningRate = hyper['LearningRate']
    batchSize = hyper['BatchSize']
    
    sequenceLength = hyper['SlideWindowSize']
    hiddenSize = hyper['HiddenLayers']
    numLayers = hyper['NumLayers']
    inputSize = param['InputDimension']
    outputSize = param['OutputDimension']
    
    datasetPath = paths['Data']
    testProportion = param['TestSetProportion']
    preloadModelFlag = param['PreloadModelFile']

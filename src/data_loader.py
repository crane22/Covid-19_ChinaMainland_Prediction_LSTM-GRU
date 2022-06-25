#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from config import param, paths, hyper

def loadData():
    dataSource = param['CurrentDataSource'].split('-')
    if dataSource[0] == "mainland":
        dictPath = paths['Data_Mainland']
    elif dataSource[0] == "province":
        dictPath = paths['Data_Province']
    else:
        print("Load data failed! Please check `CurrentDataSource` in `config.py`.")
        return
    dict = np.load(dictPath, allow_pickle=True).item()
    if dataSource[0] == "mainland":
        dataset = dict[param['CurrentDataType']]
    else:
        dataset = dict[dataSource[1]]
    boundNum = round(len(dataset) * (1 - hyper['TestSetProportion']))
    trainSet = dataset[:boundNum]
    testSet = dataset[boundNum:]
    return trainSet, testSet
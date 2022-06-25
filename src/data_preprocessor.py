#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import csv
import numpy as np
from config import paths
from config import provinces

def csvFile2List(csvFilePath):
    csvFile = open(csvFilePath, 'r')
    csvReader = csv.reader(csvFile)
    header = next(csvReader)
    date = []
    data = []
    for row in csvReader:
        if row == []:
            continue
        if len(row) == 2:
            row[0] = row[0][0:4] + '-' + row[0][4:6] + '-' + row[0][6:8]
            date.append(row[0])
            row[1] = int(row[1])
            data.append([row[1]])
        if len(row) == 4:
            row[0] = row[0][0:4] + '-' + row[0][4:6] + '-' + row[0][6:8]
            date.append(row[0])
            row[1] = int(row[1])
            row[2] = int(row[2])
            # row[3] = int(row[3])
            # data.append([row[1], row[2], row[3]])
            data.append([row[1], row[2]])
    data = np.array(data)
    return date, data

if __name__ == "__main__":
    csvFilePath = os.path.join(paths['Data_Raw'], "全国" + ".csv")
    date, data = csvFile2List(csvFilePath)
    label = ["mainland-confirmed", "mainland-recovered"]
    for p in provinces:
        pName = provinces[p]
        label.append("province-" + pName)
        csvFilePath = os.path.join(paths['Data_Raw'], p + ".csv")
        _, pData = csvFile2List(csvFilePath)
        data = np.concatenate((data, pData), axis=1)
    dataset = np.array([date, label, data], dtype=object)
    np.save(paths['Data'], dataset)
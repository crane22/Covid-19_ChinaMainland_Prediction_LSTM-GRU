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
    content = []
    for row in csvReader:
        if row == []:
            continue
        if len(row) == 2:
            row[0] = row[0][0:4] + '-' + row[0][4:6] + '-' + row[0][6:8]
            row[1] = int(row[1])
            content.append(row)
        if len(row) == 4:
            if content == []:
                content.append([])
                content.append([])
            row[0] = row[0][0:4] + '-' + row[0][4:6] + '-' + row[0][6:8]
            row[1] = int(row[1])
            row[2] = int(row[2])
            content[0].append(row[0:2])
            content[1].append(row[0:3:2])
    return content

if __name__ == "__main__":
    csvFolderPath = paths['Data_Raw']

    csvFilePath = os.path.join(csvFolderPath, "全国" + ".csv")
    mainlandList = csvFile2List(csvFilePath)
    mainlandData = {"confirmed" : mainlandList[0], 
                    "recovered" : mainlandList[1]}
    mainlandDictPath = paths['Data_Mainland']
    np.save(mainlandDictPath, mainlandData)
    
    provincesData = {}
    for p in provinces:
        pName = provinces[p]
        csvFilePath = os.path.join(csvFolderPath, p + ".csv")
        pData = csvFile2List(csvFilePath)
        provincesData.update({pName : pData})
    provinceDictPath = paths['Data_Province']
    np.save(provinceDictPath, provincesData)

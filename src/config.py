#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

hyper = {
    "SlideWindowSize" : 20,
    # "BatchSize" : 100,
}

param = {
    "RawDataFolder" : "data/raw",
    "DataFolder" : "data",
    "DataFileName" : "data.npy",
    "OutputFolder" : "output",
    "OutputModelFileName" : "model.pth",
    "PreloadModelFile" : False,
    # "ValidSetProportion" : 0.1,
    "TestSetProportion" : 0.1,
}
pathWorkingDirectory = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 1)[0]
paths = {
    "Data_Raw" : os.path.join(pathWorkingDirectory, param['RawDataFolder']),
    "Data" : os.path.join(pathWorkingDirectory, param['DataFolder'], param['DataFileName']),
    "Output_Model" : os.path.join(pathWorkingDirectory, param['OutputFolder'], param['OutputModelFileName']),
}

provinces = {
    '安徽': 'Anhui',
    '北京': 'Beijing',
    '福建': 'Fujian',
    '甘肃': 'Gansu',
    '广东': 'Guangdong',
    '广西': 'Guangxi',
    '贵州': 'Guizhou',
    '海南': 'Hainan',
    '河北': 'Hebei',
    '河南': 'Henan',
    '黑龙江': 'Heilongjiang',
    '湖北': 'Hubei',
    '湖南': 'Hunan',
    '吉林': 'Jilin',
    '江苏': 'Jiangsu',
    '江西': 'Jiangxi',
    '辽宁': 'Liaoning',
    '内蒙古': 'InnerMongolia',
    '宁夏': 'Ningxia',
    '青海': 'Qinghai',
    '山东': 'Shandong',
    '山西': 'Shanxi',
    '陕西': 'Shaanxi',
    '上海': 'Shanghai',
    '四川': 'Sichuan',
    '天津': 'Tianjin',
    '西藏': 'Tibet',
    '新疆': 'Xinjiang',
    '云南': 'Yunnan',
    '浙江': 'Zhejiang',
    '重庆': 'Chongqing',
    # '香港': 'HongKong',
    # '澳门': 'Macao',
    # '台湾': 'Taiwan',
    # '南海诸岛': 'South China Sea Islands'
}
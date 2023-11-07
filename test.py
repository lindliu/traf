#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:40:31 2023

@author: dliu
"""

import pandas as pd
import numpy as np 
# data = pd.read_csv('traffic.csv')

# data = pd.read_csv('./Datasets/PEMSd7.csv')

data = pd.read_csv('./Datasets/PeMSD7_V_228.csv')
data = pd.read_csv('./Datasets/PeMSD7_M_Station_Info.csv')



import pickle
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

adj = load_pickle('./Datasets/adj_mx.pkl')



data = pd.read_csv('./Datasets/PEMS08/PEMS08.csv')
data1 = np.load('./Datasets/PEMS08/PEMS08.npz')

num = data1['data'].shape[1]
adj_mx = np.zeros([num, num])
for i,j,cost in zip(data['from'],data['to'],data['cost']):
    adj_mx[i,j] = cost
adj_mx = adj_mx+np.eye(num)
D = np.diag(adj_mx.sum(1)**(-.5))

adj_mx = D@adj_mx@D




'''
Created on 15 May 2020

@author: oskar, mehmet
'''
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn import metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, average_precision_score
from sklearn.datasets import fetch_openml

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def P_derivation(datafr, key1, key2, key3, key4):
    P1 = datafr[key1] - datafr['T1_x']
    P2 = datafr[key2] - datafr['T1_y']
    P3 = datafr[key3] - datafr['T1_x']
    P4 = datafr[key4] - datafr['T1_y']
    return P1, P2, P3, P4

def length(P1, P2, P3, P4):
    
    vec_length_1 = abs(pow((pow(P1,2)+pow(P2,2)),1/2))
    vec_length_2 = abs(pow((pow(P3,2)+pow(P4,2)),1/2))
    return vec_length_1, vec_length_2

def angle(datafr):
    datafr.columns = ['frame_number','T1_x','T1_y','1_mx','1_my','1_ex','1_ey','2_mx','2_my','2_ex','2_ey','3_mx','3_my','3_ex','3_ey','4_mx','4_my','4_ex','4_ey','orientation','label']
    i=0
    keys = ['1_mx','1_my','1_ex','1_ey','2_mx','2_my','2_ex','2_ey','3_mx','3_my','3_ex','3_ey','4_mx','4_my','4_ex','4_ey']
    while i < 4:
        if i==0:
            key1, key2, key3, key4 = '1_ex', '1_ey', '2_ex', '2_ey'
        elif i == 1:
            key1, key2, key3, key4 = '2_ex', '2_ey', '3_ex', '3_ey'
        elif i == 2:
            key1, key2, key3, key4 = '3_ex', '3_ey', '4_ex', '4_ey'
        else:
            key1, key2, key3, key4 = '4_ex', '4_ey', '1_ex', '1_ey'
        P1, P2, P3, P4 = P_derivation(datafr,key1,key2,key3,key4)
        temp_df1 = pd.DataFrame({'A1': P1, 'A2': P2})
        temp_df2 = pd.DataFrame({'B1': P3, 'B2': P4})
        
        cosang = temp_df1['A1']*temp_df2['B1'] + temp_df1['A2']*temp_df2['B2']
        sinang = np.linalg.norm(np.cross(temp_df1, temp_df2))
        angle = np.arctan2(sinang, cosang)  
        if i==0:
            datafr['angle1'] = angle
        elif i == 1:
            datafr['angle2'] = angle
        elif i == 2:
            datafr['angle3'] = angle
        else:
            datafr['angle4'] = angle
        i=i+1
    datafr = datafr.drop(['1_mx','1_my','1_ex','1_ey','2_mx','2_my','2_ex','2_ey','3_mx','3_my','3_ex','3_ey','4_mx','4_my','4_ex','4_ey'], 1)
    return datafr

def orientation_strategy(datafr):
    datafr['angle1'] = datafr['orientation'] - datafr['angle1']
    datafr['angle2'] = datafr['orientation'] - datafr['angle2']
    datafr['angle3'] = datafr['orientation'] - datafr['angle3']
    datafr['angle4'] = datafr['orientation'] - datafr['angle4']
    datafr = datafr.drop(['orientation'],1)
    return datafr
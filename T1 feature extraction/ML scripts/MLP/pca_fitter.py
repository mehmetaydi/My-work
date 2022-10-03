'''
Created on 10 Jul 2020

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
from sklearn.model_selection import cross_val_score

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tsne_visualization
import feature_derivation
import math
from scipy import sparse

def pca_model(PCAmodel, input_train, input_test, labels_train, labels_test, solver_type):
    
    input_train = PCAmodel.transform(input_train)
    input_test = PCAmodel.transform(input_test)
    clf = MLPClassifier(solver=solver_type, alpha=1e-5, hidden_layer_sizes=(9,12,6,), random_state=0, max_iter=20000, shuffle=True)   
    clf.fit(input_train, labels_train)
    predicted = clf.predict(input_test)
    conf_matrix = metrics.confusion_matrix(labels_test,predicted)
    tn, fp, fn, tp = metrics.confusion_matrix(labels_test, predicted).ravel()
    scores = cross_val_score(clf, input_test, labels_test, cv=5)
        
    print('Fitting of training data complete.')
    print('Predicting based on test data.')
    print("\nTraining set score: %f" % clf.score(input_train, labels_train))
    print("Test set score: %f" % clf.score(input_test, labels_test))
    print('Confusion matrix: \n')
    print('TN:', tn)
    print('TP: ',tp)
    print('FN: ',fn)
    print('FP:' ,fp)
    print('CROSS VALIDATION SCORES:', scores)
    print("\n------------------------------------------------------------") 
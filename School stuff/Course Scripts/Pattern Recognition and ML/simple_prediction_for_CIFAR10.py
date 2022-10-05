# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:40:12 2020

@author: mehmet
"""


import numpy
import pickle
import numpy as np
# from tqdm import tqdm
import random
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

data_batch_1 = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/data_batch_5')

trdata =np.concatenate(( data_batch_1["data"],data_batch_2["data"],data_batch_3["data"],data_batch_4["data"],data_batch_5["data"]))
trlabel = np.concatenate(( data_batch_1["labels"],data_batch_2["labels"],data_batch_3["labels"],data_batch_4["labels"],data_batch_5["labels"]))
trlabel = np.array(trlabel)


datadict = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/test_batch')

X_test = datadict["data"]
x = datadict["labels"]

labeldict = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
x = np.array(x)

def cifar10_classier_random(x):
    

    Accuracy  = 100*np.mean( random_label == x )
    
    return  Accuracy

random_label = random.choice(x)
Accuracy = cifar10_classier_random(x)
print(f'Accuracy of the system is {Accuracy}')
# Accuracy  = 100*np.mean( random_label == x )


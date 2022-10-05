# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:46:40 2020

@author: mehmet
"""
import numpy
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import norm ,multivariate_normal
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean

# from skimage.transform import rescale, resize, downscale_local_mean
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
x = trdata
y = datadict["data"]
test_label = datadict["labels"]
test_label = np.array(test_label)
y = y.reshape(10000, 3, 32, 32)
labeldict = unpickle(r'C:\Users\mehmet\Desktop\Master at Machine Learning\1. Period\Introduction to pattern recognition and ML\Exercise 2\cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
label_names = np.array(label_names)
x = x.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
# Average color for test data
def cifar10_color(x): 
    Xf = []
    for i in range(0,len(trlabel)):
        average1 = x[i].mean(axis=0).mean(axis=0)        
        Xf.append(average1)
    Xf= np.array(Xf)
    return Xf      
Yf = [] # avarage color for test data
for i in range(0,len(y)):
    average1 = y[i].mean(axis=0).mean(axis=0) 
    Yf.append(average1)
Yf =np.array(Yf)      
Xf = cifar10_color(x)    
Y = [ Xf[:,0],Xf[:,1],Xf[:,2],trlabel]
Y = np.array(Y)
Y = Y.transpose()  
## to find sigma and mean values 
def cifar_10_naivebayes_learn(Xf,Y):
    mu = []
    sigma = []
    for i in range(0,len(label_names)):  
        a = [np.mean(Y[Y[:,3]==i, 0:], axis=0)] 
        a2 = [np.std(Y[Y[:,3]==i, 0:], axis=0)] 
        sigma.append(a2)
        mu.append(a)     
    sigma = np.array(sigma)    
    mu = np.array(mu)     
    Y = mu.reshape(10,4) , sigma.reshape(10,4)         
    return Y        
mu,sigma = cifar_10_naivebayes_learn(Xf,Y)   
sigma = sigma[:,0:3] 
mu =  mu[:,0:3] 
p = 0.1 + np.zeros([10,1])
####################     Exercise 3.1      ####################################
def cifar10_classier_naivebayes(x,mu,sigma,p):    
    predicted_class = []
    for i in tqdm(range(0,len(y))):
        P = []
        for k in range(0,len(mu)):
            denum = 0
            denum = ((norm.pdf(Yf[i,0],sigma[k,0],mu[k,0])*norm.pdf(Yf[i,1],sigma[k,1],mu[k,1])*norm.pdf(Yf[i,2],sigma[k,2],mu[k,2])))*p[k]
            denum += denum
        for k in range(0,len(mu)):    
            P1 =  ((norm.pdf(Yf[i,0],mu[k,0],sigma[k,0])*norm.pdf(Yf[i,1],mu[k,1], sigma[k,1])*norm.pdf(Yf[i,2], mu[k,2], sigma[k,2]))*p[k])/denum                  
            P.append(P1)
        predicted_class1 = P.index(max(P))
        predicted_class.append(predicted_class1)
    predicted_class = np.array(predicted_class)
    return predicted_class
predicted_class = cifar10_classier_naivebayes(x,mu,sigma,p)
Accuracy = 100*np.mean( predicted_class == test_label )
print(f'Accuracy of the system for Exercise 3.1 is {Accuracy} %')

# ####################     Exercise 3.2      ####################################
covariance =[]
for i in range(0,10):
    class_imgs = Xf[np.where(trlabel==i)]    
    b= np.cov(class_imgs, rowvar= False)
    covariance.append(b)
covariance = np.array(covariance)    
def cifar10_classier_naivebayes(x,mu,covariance,p):    
    predicted_class = []    
    for i in tqdm(range(0,len(test_label))):
        P2 = []
        for k in range(0,len(mu)):
            denum = 0
            denum =(multivariate_normal.pdf(Yf[i],mu[k], covariance[k]))*p[k]
            denum += denum
        for k in range(0,len(mu)):             
            P1 =(multivariate_normal.pdf(Yf[i],mu[k],covariance[k])*p[k])/denum 
            P2.append(P1)         
        predicted_class1 = P2.index(max(P2)) 
        predicted_class.append(predicted_class1)          
    predicted_class = np.array(predicted_class)
    return predicted_class
predicted_class = cifar10_classier_naivebayes(x,mu,covariance,p)
Accuracy = 100*np.mean( predicted_class == test_label )
print(f'Accuracy of the system for Exercise 3.2 is {Accuracy} % ')


# ####################     Exercise 3.3      ####################################

for size in tqdm([1,2,4,6]):
    rgb_vals = []
    for i in range(y.shape[0]):
        # Convert images to mean values of each color channel
        img = y[i]
        img_8x8 = resize(img, (size, size))               
        r_vals = img_8x8[:,:,0].reshape(size*size)
        g_vals = img_8x8[:,:,1].reshape(size*size)
        b_vals = img_8x8[:,:,2].reshape(size*size)
        rgb_vals1 = np.concatenate((r_vals,g_vals,b_vals)) 
        rgb_vals.append(rgb_vals1)
    rgb_vals = np.array(rgb_vals)
    X_mean = []
    covariance =[]
    for i in range(0,10):
        class_imgs = rgb_vals[np.where(test_label==i)] 
        X_mean1 = np.mean(class_imgs, axis=0)
        covariance1= np.cov(class_imgs, rowvar= False)
        X_mean.append(X_mean1)
        covariance.append(covariance1)
    X_mean = np.array(X_mean)
    covariance = np.array(covariance)
    
    def cifar10_classier_naivebayes(x,mu,covariance,p):    
        predicted_class = []    
        for i in range(0,len(test_label)):
            P2 = []
            for k in range(0,len(p)):
                denum = 0
                denum =(multivariate_normal.pdf(rgb_vals[i],X_mean[k], covariance[k]))*p[k]
                denum += denum
            for k in range(0,len(p)):             
                P1 =(multivariate_normal.pdf(rgb_vals[i],X_mean[k],covariance[k])*p[k])/denum 
                P2.append(P1)         
            predicted_class1 = P2.index(max(P2)) 
            predicted_class.append(predicted_class1)          
        predicted_class = np.array(predicted_class)
        return predicted_class
    predicted_class = cifar10_classier_naivebayes(x,mu,covariance,p)
    Accuracy = 100*np.mean( predicted_class == test_label )
    # print(f'Accuracy of the system is {Accuracy} % ')
    plt.plot(size, Accuracy, 'ro' )
    plt.xlabel('size')
    plt.ylabel('Accuracy')
    
########################## THE END ####################################################






























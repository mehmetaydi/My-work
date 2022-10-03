# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:37:13 2021

@author: phmeay
"""

from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib 

dataK5D2 = pd.read_csv('dataK5D2.csv') # read the file dataK5D2

dataK5D3 = pd.read_csv('dataK5D3.csv') # read the file dataK5D3

del dataK5D2["LABEL"] # dropping label column

del dataK5D3["LABEL"] # dropping label column

def elbow_method(data,file_name):

    x = data.to_numpy() # converting pandas data frame to numpy array for visualization
    distorsions = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distorsions.append(kmeans.inertia_)
    
    fig = plt.figure(figsize=(10, 10))
    plt.plot(range(2, 10), distorsions,c ='purple',linewidth=4, marker='o')
    plt.ylabel('Cost')
    plt.xlabel('Amount of cluster')

    
    plt.grid(True)
    plt.title('The Elbow Method showing the optimal k for {}'.format(file_name))
    if len(x[1,:]) ==2: # 2D data
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(x[:,0],x[:,1],c ='r')
        plt.title('2D representation of dataK5D2 ')
        
    if len(x[1,:]) ==3: # 3D data
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        
        x_coord = x[:,0]
        y_coord = x[:,1]
        z_coord = x[:,2]
        
        ax.set_title('3D representation of dataK5D3 ')
        ax.scatter(x_coord, y_coord, z_coord,c ='r')
        
        plt.show()


elbow_method(dataK5D2,'dataK5D2') # visualing

elbow_method(dataK5D3,'dataK5D3')




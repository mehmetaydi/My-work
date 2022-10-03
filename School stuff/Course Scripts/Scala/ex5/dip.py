# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:37:13 2021

@author: phmeay
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

%matplotlib 
dataK5D2 = pd.read_csv('dataK5D2.csv') # read the file dataK5D2

dataK5D3 = pd.read_csv('dataK5D3.csv') # read the file dataK5D3

del dataK5D2["LABEL"] # dropping label column

del dataK5D3["LABEL"] # dropping label column

# Import ElbowVisualizer
k =5

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

X = dataK5D3
x = X.to_numpy() # converting pandas data frame to numpy array for visualization
distorsions = []
for k in range(2, 50):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(2, 50), distorsions)
plt.ylabel('Distortion')
plt.xlabel('K')

plt.grid(True)
plt.title('The Elbow Method showing the optimal k for dataK5D3')



Y = dataK5D2
y =Y.to_numpy() # converting pandas data frame to numpy array for visualization
distorsions = []
for k in range(2, 50):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Y)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(2, 50), distorsions)
plt.ylabel('Distortion')
plt.xlabel('K')

plt.grid(True)
plt.title('The Elbow Method showing the optimal k for dataK5D2')



fig = plt.figure(figsize=(10, 10))
plt.scatter(y[:,0],y[:,1])
plt.title('2D representation of dataK5D2 ')



from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure(figsize=(10, 10))
ax = Axes3D(fig)

x_coord = x[:,0]
y_coord = x[:,1]
z_coord = x[:,2]

ax.set_title('3D representation of dataK5D3 ')
ax.scatter(x_coord, y_coord, z_coord)

pyplot.show()

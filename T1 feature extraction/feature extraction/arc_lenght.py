
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:03:50 2020

@author: mehmet
"""
'''
this file contains two different sctripts that find the Arc lenght between 
start points and end points the next one(start from Line 83 ) is for finding the arc lenght
between start points and mid points. if you run this code do not forget to run Imputer code for 
removing all nan values in dataset. May God be with us :) 

'''
import numpy as np
import math


data=np.genfromtxt(r'C:\Users\mehmet\Desktop\dataset\dataset_04_1.txt', delimiter='  ', dtype=int)

data1=np.genfromtxt(r'C:\Users\mehmet\Desktop\dataset\dataset_03.txt', delimiter='  ', dtype=float)

# ############### This code is for calculating the arc lenght of the start points and end points of the vertexs #############


chord_length = []
angles = []
for i in range(0,len(data)):
    d1 =  math.sqrt((data[i,1]-data[i,5])**2 + (data[i,2]-data[i,6])**2 )  
    d2 =  math.sqrt((data[i,1]-data[i,9])**2 + (data[i,2]-data[i,10])**2 )  
    d3 =  math.sqrt((data[i,1]-data[i,13])**2 + (data[i,2]-data[i,14])**2 )  
    d4 =  math.sqrt((data[i,1]-data[i,17])**2 + (data[i,2]-data[i,18])**2 )
    alpha1 = np.arctan(abs(data[i,2]-data[i,6])/abs(data[i,1]-data[i,5]))*180/np.pi
    alpha2 = 180-(np.arctan((abs(data[i,2]-data[i,10])/abs(data[i,1]-data[i,9])))*180/np.pi)
    alpha3 = 270-(np.arctan((abs(data[i,2]-data[i,14])/abs(data[i,1]-data[i,13])))*180/np.pi)
    alpha4 = 360-(np.arctan((abs(data[i,2]-data[i,18])/abs(data[i,1]-data[i,17])))*180/np.pi)
    chord_length.append((d1,d2,d3,d4))
    angles.append((alpha1,alpha2,alpha3,alpha4))
    
chord_length = np.array(chord_length) 

angles = np.array(angles)


teta = []
for a in range(0,len(data1)):
    teta1 = 2*np.arcsin((chord_length[a,0]/2)*data1[a,7])*180/np.pi
    teta2 = 2*np.arcsin((chord_length[a,1]/2)*data1[a,8])*180/np.pi
    teta3 = 2*np.arcsin((chord_length[a,2]/2)*data1[a,9])*180/np.pi
    teta4 = 2*np.arcsin((chord_length[a,3]/2)*data1[a,10])*180/np.pi
    teta.append((teta1,teta2,teta3,teta4))
    
teta = np.array(teta)

Arc_length = []
heigh = []
for b in range(0,len(data1)):
    Arc_length1 = teta[b,0]*(1/data1[b,7])*np.pi/180 
    Arc_length2 = teta[b,1]*(1/data1[b,8])*np.pi/180 
    Arc_length3 = teta[b,2]*(1/data1[b,9])*np.pi/180 
    Arc_length4 = teta[b,3]*(1/data1[b,10])*np.pi/180
    h1 = np.sqrt(((1/data1[b,7])**2)- (((chord_length[b,0])**2)/4))
    h2 = np.sqrt(((1/data1[b,8])**2)- (((chord_length[b,1])**2)/4))
    h3 = np.sqrt(((1/data1[b,9])**2)- (((chord_length[b,2])**2)/4))
    h4 = np.sqrt(((1/data1[b,10])**2)- (((chord_length[b,3])**2)/4))
    heigh.append((h1,h2,h3,h4))
    Arc_length.append((Arc_length1,Arc_length2,Arc_length3,Arc_length4))
Arc_length = np.array(Arc_length) 
heigh = np.array(heigh)

Area = []
for t in range(0,len(data1)):
    A1 = ((np.pi*((1/data1[t,7])**2)*abs(teta[t,0])/360) - ((chord_length[t,0]/2)*heigh[t,0]))
    A2 = ((np.pi*((1/data1[t,8])**2)*abs(teta[t,1])/360) - ((chord_length[t,1]/2)*heigh[t,1]))
    A3 = ((np.pi*((1/data1[t,9])**2)*abs(teta[t,2])/360) - ((chord_length[t,2]/2)*heigh[t,2]))
    A4 = ((np.pi*((1/data1[t,10])**2)*abs(teta[t,3])/360) - ((chord_length[t,3]/2)*heigh[t,3]))
    Area.append((A1,A2,A3,A4))
Area = np.array(Area)
dataset = []
for k in range(0,len(data1)):
    
    (dataset.append(( data1[k,0],data1[k,1],data1[k,2],
                      angles[k,0],angles[k,1],angles[k,2],angles[k,3],
                      data1[k,7],data1[k,8],data1[k,9],data1[k,10]
                      ,chord_length[k,0],chord_length[k,1],chord_length[k,2]
                      ,chord_length[k,3],teta[k,0],teta[k,1],teta[k,2],teta[k,3],
                      Arc_length[k,0],Arc_length[k,1],Arc_length[k,2],Arc_length[k,3],
                      Area[k,0],Area[k,1] ,Area[k,2],Area[k,3], 
                      data1[k,11],data1[k,12] )))

dataset = np.array(dataset) 

np.savetxt('test_start_end_points.txt', dataset, delimiter='   ')   # X is an array


###########################################################################

############### This code is for calculating the arc lenght of the start points and mid points of the vertexs #############

chord_length_mid = []
angles_mid = []
for i in range(0,len(data)):
    d1_mid =  math.sqrt((data[i,1]-data[i,3])**2 + (data[i,2]-data[i,4])**2 )  
    d2_mid =  math.sqrt((data[i,1]-data[i,7])**2 + (data[i,2]-data[i,8])**2 )  
    d3_mid =  math.sqrt((data[i,1]-data[i,11])**2 + (data[i,2]-data[i,12])**2 )  
    d4_mid =  math.sqrt((data[i,1]-data[i,15])**2 + (data[i,2]-data[i,16])**2 )
    alpha1_mid = np.arctan(abs(data[i,2]-data[i,4])/abs(data[i,1]-data[i,3]))*180/np.pi
    alpha2_mid = 180-(np.arctan((abs(data[i,2]-data[i,8])/abs(data[i,1]-data[i,7])))*180/np.pi)
    alpha3_mid = 270-(np.arctan((abs(data[i,2]-data[i,12])/abs(data[i,1]-data[i,11])))*180/np.pi)
    alpha4_mid = 360-(np.arctan((abs(data[i,2]-data[i,16])/abs(data[i,1]-data[i,15])))*180/np.pi)
    chord_length_mid.append((d1_mid,d2_mid,d3_mid,d4_mid))
    angles_mid.append((alpha1_mid,alpha2_mid,alpha3_mid,alpha4_mid))
    
chord_length_mid = np.array(chord_length_mid) 

angles_mid = np.array(angles_mid)


teta_mid = []
for a in range(0,len(data1)):
    teta1_mid = 2*np.arcsin((chord_length_mid[a,0]/2)*data1[a,7])*180/np.pi
    teta2_mid = 2*np.arcsin((chord_length_mid[a,1]/2)*data1[a,8])*180/np.pi
    teta3_mid = 2*np.arcsin((chord_length_mid[a,2]/2)*data1[a,9])*180/np.pi
    teta4_mid = 2*np.arcsin((chord_length_mid[a,3]/2)*data1[a,10])*180/np.pi
    teta_mid.append((teta1_mid,teta2_mid,teta3_mid,teta4_mid))
    
teta_mid = np.array(teta_mid)
heigh_mid = []
Arc_length_mid = []
for b in range(0,len(data1)):
    Arc_length1_mid = teta_mid[b,0]*(1/data1[b,7])*np.pi/180 
    Arc_length2_mid = teta_mid[b,1]*(1/data1[b,8])*np.pi/180 
    Arc_length3_mid = teta_mid[b,2]*(1/data1[b,9])*np.pi/180 
    Arc_length4_mid = teta_mid[b,3]*(1/data1[b,10])*np.pi/180 
    h1_mid = np.sqrt(((1/data1[b,7])**2)- (((chord_length_mid[b,0])**2)/4))
    h2_mid = np.sqrt(((1/data1[b,8])**2)- (((chord_length_mid[b,1])**2)/4))
    h3_mid = np.sqrt(((1/data1[b,9])**2)- (((chord_length_mid[b,2])**2)/4))
    h4_mid = np.sqrt(((1/data1[b,10])**2)- (((chord_length_mid[b,3])**2)/4))
    heigh_mid.append((h1_mid,h2_mid,h3_mid,h4_mid))
    Arc_length_mid.append((Arc_length1_mid,Arc_length2_mid,Arc_length3_mid,Arc_length4_mid))
Arc_length_mid = np.array(Arc_length_mid) 

Area_mid = []
for t in range(0,len(data1)):
    A1_mid = ((np.pi*((1/data1[t,7])**2)*abs(teta_mid[t,0])/360) - ((chord_length_mid[t,0]/2)*heigh[t,0]))
    A2_mid  = ((np.pi*((1/data1[t,8])**2)*abs(teta_mid[t,1])/360) - ((chord_length_mid[t,1]/2)*heigh[t,1]))
    A3_mid  = ((np.pi*((1/data1[t,9])**2)*abs(teta_mid[t,2])/360) - ((chord_length_mid[t,2]/2)*heigh[t,2]))
    A4_mid  = ((np.pi*((1/data1[t,10])**2)*abs(teta_mid[t,3])/360) - ((chord_length_mid[t,3]/2)*heigh[t,3]))
    Area_mid.append((A1_mid ,A2_mid ,A3_mid ,A4_mid ))
Area_mid = np.array(Area_mid)


dataset_mid = []


for k in range(0,len(data1)):
    
    (dataset_mid.append((data1[k,0],data1[k,1],data1[k,2],
                      angles_mid[k,0],angles_mid[k,1],angles_mid[k,2],angles_mid[k,3],
                      data1[k,7],data1[k,8],data1[k,9],data1[k,10]
                      ,chord_length_mid[k,0],chord_length_mid[k,1],chord_length_mid[k,2]
                      ,chord_length_mid[k,3],teta_mid[k,0],teta_mid[k,1],teta_mid[k,2],teta_mid[k,3],
                      Arc_length_mid[k,0],Arc_length_mid[k,1],Arc_length_mid[k,2],Arc_length_mid[k,3],
                      Area_mid[k,0],Area_mid[k,1] ,Area_mid[k,2],Area_mid[k,3], 
                      data1[k,11],data1[k,12])))

dataset_mid = np.array(dataset_mid) 

np.savetxt('test_for_mid_points.txt', dataset_mid, delimiter='   ')   # X is an array



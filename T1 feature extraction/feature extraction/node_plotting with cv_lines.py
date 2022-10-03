# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:34:41 2020

@author: mehmet
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy
import cv2

'''
Please adjust the start, stop and step points before running (Line 33)
for example if your image number starts with 301 and ends with 499 
then select the start point as 301 and stop as 500 
'''
mypath=r"C:\Users\mehmet\Desktop\Machine Learning\images"

data=np.genfromtxt(r'C:\Users\mehmet\Desktop\Machine Learning\test_set_01.txt', delimiter='  ', dtype=int)

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

images = numpy.empty(len(onlyfiles), dtype=object)

b = np.arange(start=301, stop=500, step=1, dtype=int)

for n in range(1,len(onlyfiles)+1):
    
    images[n-1] = cv2.imread( join(mypath,onlyfiles[n-1]))
    
    for i in range(1, len(data)+1):
        
        if data[i-1,0] == b[n-1,] :
            
            if data[i-1,8]==1 :
          
           
 
               cv2.circle(images[n-1], (data[i-1,1],data[i-1,2]),3,(255,255,255),-1)
                   
               cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,3]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,3]*math.pi/180))),(255),3)
               
               cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,4]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,4]*math.pi/180))),(255),3)
               
               cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,5]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,5]*math.pi/180))),(255),3)
               
               cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,6]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,6]*math.pi/180))),(255),3)
            else:
                
                cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,3]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,3]*math.pi/180))),(255),3)
                
                cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,4]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,4]*math.pi/180))),(255),3)
                
                cv2.line(images[n-1], (data[i-1,1],data[i-1,2]),(int(data[i-1,1]+ 15*math.cos(data[i-1,5]*math.pi/180)),int(data[i-1,2]+ 15*math.sin(data[i-1,5]*math.pi/180))),(255),3)
                           
                k = cv2.imwrite("im000{}".format(b[n-1])+'.jpg', images[n-1])
                
               
               # cv2.imshow("im000{}".format(b[n-1]), images[n-1])
             
        
cv2.waitKey(1) 
cv2.destroyAllWindows() 
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:39:05 2020
@author: mehmet
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy


def plot_nodes(mypath, filename,b,data,t,unique, parameters = None):
    
    '''
    mypath is path to dataset where are the images in
    filename is path to output folder where the  images with red cross go
    b is a virtual array to get correct curvature on top of images
    Unique is used for removing duplicates in data file in order to compare image and text file smootly

    '''
    
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    
    images = numpy.empty(len(onlyfiles), dtype=object)
    
     
    for k in range(1,len(b)+1):
        
        for n in range(1,len(onlyfiles)+1):
             
                if b[k-1,] ==  unique[n-1,]:
            
                    dpi = 30
                        
                    images[n-1] = plt.imread( join(mypath,onlyfiles[n-1]))
                        
                    num_rows, num_cols,RGB = images[n-1].shape
                        
                    # What size does the figure need to be in inches to fit the image?
                    figsize = num_rows / float(dpi), num_cols / float(dpi)
                        
                    # Create a figure of the right size with one axes that takes up the full figure
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_axes([0, 0, 1, 1])
                        
                    ax.axis('off')
                        
                    ax.set(xlim=[-0.5, num_rows - 0.5], ylim=[num_cols - 0.5, -0.5], aspect=1)
                    for i in range(1,len(data)+1):
                            
                        if data[i-1,0]== b[k-1,]:
                            
                            def getCurve(x0,y0,angle,curvature):
                                # t = np.arange(25)
                                x = t
                                y = curvature * t*t
                                
                                x_new = x*math.cos(angle*np.pi/180) - y*math.sin(angle*np.pi/180) + x0
                                y_new = x*math.sin (angle*np.pi/180) + y*math.cos (angle*np.pi/180) + y0
                                
                                plt.plot(x_new, y_new, 'r', linewidth = 6)
                                    
                                   
                            plt.imshow(images[n-1], getCurve(data[i-1,1],data[i-1,2],data[i-1,3],data[i-1,7]))
                            plt.imshow(images[n-1], getCurve(data[i-1,1],data[i-1,2],data[i-1,4],data[i-1,8]))
                            plt.imshow(images[n-1], getCurve(data[i-1,1],data[i-1,2],data[i-1,5],data[i-1,9]))
                            plt.imshow(images[n-1], getCurve(data[i-1,1],data[i-1,2],data[i-1,6],data[i-1,10]))
                            if int(data[i-1,0])<10:
                                fig.savefig("im00000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True) 
                                    
                            elif 10<=int(data[i-1,0])<100:
                                fig.savefig("im0000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)  
                                    
                            elif 100<=int(data[i-1,0])<1000:
                                fig.savefig("im000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)    
                                    
                            else:
                                fig.savefig("im00{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)    
                             

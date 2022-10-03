# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:05:03 2020

@author: mehmet
"""

import numpy 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from os import listdir
from os.path import isfile, join

dpi = 30
data=np.genfromtxt(r'C:\Users\mehmet\Desktop\dataset\dataset_04_1.txt', delimiter='  ', dtype=float)
unique = np.unique(data[:,0], axis = 0 ) 

mypath=r"C:\Users\mehmet\Desktop\images"

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

images = numpy.empty(len(onlyfiles), dtype=object)

b = np.arange(start=20, stop=50, step=1, dtype=int)
# data = data.reshape(1,-1)
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
   
                            X, y ,X1, y1 =  data[i-1,[1,3,5]], data[i-1,[2,4,6]] ,data[i-1,[1,7,9]],data[i-1,[2,8,10]] 
                            
                            X2,y2,X3,y3 = data[i-1,[1,11,13]], data[i-1,[2,12,14]],data[i-1,[1,15,17]],data[i-1,[2,16,18]]

                            X, y ,X1, y1 = X.reshape(-1,1),y.reshape(-1,1),X1.reshape(-1,1),y1.reshape(-1,1) 
                            
                            X2,y2,X3,y3 = X2.reshape(-1,1),y2.reshape(-1,1),X3.reshape(-1,1),y3.reshape(-1,1)                        
                           #######################################################################################
                            X4,y4,X5,y5 =  data[i-1,[1,3,5]],data[i-1,[2,4,6]],data[i-1,[1,7,9]],data[i-1,[2,8,10]]
                            
                            X6,y6,X7,y7 = data[i-1,[1,11]],data[i-1,[2,12]],data[i-1,[1,15]],data[i-1,[2,16]]

                            X4,y4,X5,y5 = X4.reshape(-1,1),y4.reshape(-1,1),X5.reshape(-1,1),y5.reshape(-1,1)
                            
                            X6,y6,X7,y7 = X6.reshape(-1,1),y6.reshape(-1,1),X7.reshape(-1,1),y7.reshape(-1,1)
                           
                            
                            if (data[i-1,12]< data[i-1,2] and data[i-1,14]< data[i-1,12]
                               or data[i-1,2]< data[i-1,12] and data[i-1,12]< data[i-1,14]):
                
                                def Polynomial_Regression():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y)                                   
                                    plt.plot(X,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Red')
                                Polynomial_Regression()         
                                def Polynomial_Regression1():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X1)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y1)
                                    plt.plot(X1,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Blue')
                                Polynomial_Regression1()        
                                def Polynomial_Regression2():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X2)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y2)
                                    plt.plot(X2,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Green')
                                Polynomial_Regression2()    
                                def Polynomial_Regression3():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X3)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y3)
                                    plt.plot(X3,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Black')          
                                Polynomial_Regression3() 

                            else:                                 
                                def Polynomial_Regression():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X4)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y4)
                                    plt.plot(X4,pol_reg.predict(X_poly) , Linewidth = 4,color = 'Red')
                                Polynomial_Regression()         
                                def Polynomial_Regression1():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X5)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y5)
                                    plt.plot(X5,pol_reg.predict(X_poly) , Linewidth = 4,color = 'Blue')
                                Polynomial_Regression1()        
                                def Polynomial_Regression2():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X6)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y6)
                                    plt.plot(X6,pol_reg.predict(X_poly) , Linewidth = 4,color = 'Green')
                                Polynomial_Regression2()    
                                def Polynomial_Regression3():
                                    poly_reg = PolynomialFeatures(degree=7)
                                    X_poly = poly_reg.fit_transform(X7)
                                    pol_reg = LinearRegression()
                                    pol_reg.fit(X_poly, y7)
                                    plt.plot(X7,pol_reg.predict(X_poly) , Linewidth = 4,color = 'Black')                               
                                Polynomial_Regression3()
                            plt.imshow(images[n-1],Polynomial_Regression(),Polynomial_Regression1(),Polynomial_Regression2(), Polynomial_Regression3())
                                                
                            if int(data[i-1,0])<10:
                                fig.savefig("im00000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True) 
                                                
                            elif 10<=int(data[i-1,0])<100:
                                fig.savefig("im0000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)  
                                                
                            elif 100<=int(data[i-1,0])<1000:
                                fig.savefig("im000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)    
                                                
                            elif 1000<=int(data[i-1,0])<10000:
                                fig.savefig("im00{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)
                                            
                            else:
                                fig.savefig("im0{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)
                                                            
                                        
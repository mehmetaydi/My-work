
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:50:26 2020
@author: mehmet
This script is created for predicting the changing of Turkish lira against USD
For this purpose Linear regression and Polynomial regression has been tested.
The sum of the Total error has been calculated until degree 100 (Line 92) 
after this calculation  the best fitted degree has been found which is number 5. 
To improve this script i will  try to figure out how to predict for the rest of the year 2020
"""
print(__doc__)

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_excel('USD against Turkish lira.xlsx', delimiter = ',', dtype = float)

data = data.fillna(method = 'ffill')

y = np.array(data['Price'])

x= np.arange(1,263)

X = x.reshape(262,1)

y = y.reshape(262,1)
print(sum(y))
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
Lineear_Predict =lin_reg.predict(X)

# Visualizing the Linear Regression results
def Get_Linear():
    plt.scatter(X, y, color='Red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('The prediction of the Turkish liras against USD (Linear Regression)')
    plt.xlabel('Day')
    plt.ylabel('The value of the currency')
    plt.show()
    return
Get_Linear()
Linear_Accuracy = []
Linear_Error = 0
for i in range(0,len(X)):
   
    Linear_Error = Linear_Error + abs(float(y[i])-float(Lineear_Predict[i]))
    

Linear_Accuracy.append(Linear_Error)
Linear_Error = 0
    
print(sum(Linear_Accuracy))
print(np.mean(y))
Linear_Accuracy = 100 - (Linear_Accuracy/(len(X)*np.mean(y)))*100

# Fitting Polynomial Regression to the dataset
for i in [2,3,5,25]:
    poly_reg = PolynomialFeatures(degree=i)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    Poly_Predict = pol_reg.predict(X_poly)
    

    plt.scatter(X, y, color='Gold')
    
    plt.plot(X,pol_reg.predict(X_poly) , Linewidth = 2)
    
   
    plt.legend(["Polynomial degree 2","Polynomial degree 3","Polynomial degree 5","Polynomial degree 25"])
plt.title('Estimated value of the Turkish lira against USD')
plt.xlabel('Days of the year(Only weekdays) ')
plt.ylabel('Current value of exchange(₺)')
plt.savefig('Estimated value of the Turkish lira against the USD')
plt.show()


polynomial_error = 0    
Total_Error =[]    
Accuracy = 0
Total_Accuracy = []
# this loop is used for finding the best polynomial degree for the regression
# we are able to sum all the Error when the degre is 1,2,3 ... etc. 
# then  we can detect the best degree for the regression 
for k in range(0,100):
    poly_reg = PolynomialFeatures(degree=k+1)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    b = pol_reg.predict(X_poly)
    
    for p in range(0,len(X_poly)):
        # The sum of squares between the real currency values(y) and the predicted values(X_poly)
        polynomial_error = polynomial_error + (float(y[p])-float(pol_reg.predict(X_poly)[p]))**2
        Accuracy = Accuracy + abs(float(y[p])-float(pol_reg.predict(X_poly)[p]))
        
    Accuracy = 100- (Accuracy/(len(X)*np.mean(y)))*100
    Total_Accuracy.append(Accuracy)
    Accuracy = 0 
    # print(polinomial_error1)
    Total_Error.append(polynomial_error)
    polynomial_error = 0
        

Index = Total_Error.index(min(Total_Error))

print(f"The best polynomial degree is {Index}")

Error = Total_Error[Index]


print(f"The total error for degree {Index} is {Error}")

Accuracy_index = Total_Accuracy.index(max(Total_Accuracy))

Accuracy_value =  Total_Accuracy[Index]

print(f"The Accuracy of the system is {Accuracy_value}")













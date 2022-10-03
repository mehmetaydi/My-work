# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:58:07 2020

@author: mehmet
"""

'''
This code is used for Excercise 1 question 3
'''
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib qt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 3])
ax.set_ylim([0, 2])

# Enter the number of clicks that we want to see
i = input('Enter number of dots : ')

#  Add a point by left clicking
#  Remove the most recently added point by clicking mid button.
# Stop adding points and go through with recently added points by right clicking.
clicks = plt.ginput(n=int(i), timeout=200, show_clicks=True, mouse_add=1, mouse_pop=2, mouse_stop=3)

clicks = np.array(clicks)

x = clicks[:,0]
y = clicks[:,1]

print('X coordinates :', x)
print('Y coordinates :', y)

def my_linfit(x,y):
   
    a = (len(x)*sum(x*y)-sum(x)*sum(y))/(len(x)*sum(x*x)-sum(x)*sum(x))
    b = (sum(y)*sum(x*x)-sum(x)*sum(x*y))/(len(x)*sum(x*x)-(sum(x)*sum(x)))
    
    return a,b

a,b = my_linfit(x,y)
plt.plot(x,y,'kx')
xp = np.arange(-2,5,(5-(-2)/len(x)))
plt.plot(xp,a*xp+b,'r-')
print(f"My fit: a={a} and b={b}")
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:43:44 2020

@author: mehmet
"""
import numpy as np
import math

test=np.genfromtxt(r'C:\Users\mehmet\Desktop\CSM\test_for_mid_points.txt', delimiter='  ', dtype=float)
test2=np.genfromtxt(r'C:\Users\mehmet\Desktop\CSM\test_start_end_points.txt', delimiter='  ', dtype=float)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer = imputer.fit(test)
test = imputer.transform(test)

np.savetxt('test_for_mid_points.txt', test, delimiter='   ')   # X is an array

imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer = imputer.fit(test2)
test2 = imputer.transform(test2)

np.savetxt('test_start_end_points.txt', test2, delimiter='   ')   # X is an array

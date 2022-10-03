'''
Created on 13 Jul 2020

@author: oskar, mehmet
'''
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn import metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, average_precision_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math



def orientation_angles(df):
    not_done = False
    print('\n- Manipulating angles according to orientation. -')
    i = 0
    alfa_done, mid_done, theta_done = False, False, False
    while not_done == False:
        angle_type = input('\nWhich angle do you want to standardize? Type either "alfa", "mid", "theta" or "all": ')
        if angle_type == 'alfa' and alfa_done == False:
            print('Angles 1, 2, 3 and 4 are transformed.')
            df['angle1'] = df['orientation'] - df['angle1']
            df['angle2'] = df['orientation'] - df['angle2']
            df['angle3'] = df['orientation'] - df['angle3']
            df['angle4'] = df['orientation'] - df['angle4']
            alfa_done = True
        elif angle_type == 'mid' and mid_done == False:
            print('Angles 1, 2, 3 and 4 are transformed.')
            df['mid1'] = df['orientation'] - df['mid1']
            df['mid2'] = df['orientation'] - df['mid2']
            df['mid3'] = df['orientation'] - df['mid3']
            df['mid4'] = df['orientation'] - df['mid4']
            mid_done = True
        elif angle_type == 'theta' and theta_done == False:
            print('Angles 1, 2, 3 and 4 are transformed.')
            df['theta1'] = df['orientation'] - df['theta1']
            df['theta2'] = df['orientation'] - df['theta2']
            df['theta3'] = df['orientation'] - df['theta3']
            df['theta4'] = df['orientation'] - df['theta4']
            theta_done = True
        elif angle_type == 'all' and i == 0:
            df['angle1'] = df['orientation'] - df['angle1']
            df['angle2'] = df['orientation'] - df['angle2']
            df['angle3'] = df['orientation'] - df['angle3']
            df['angle4'] = df['orientation'] - df['angle4']
            df['mid1'] = df['orientation'] - df['mid1']
            df['mid2'] = df['orientation'] - df['mid2']
            df['mid3'] = df['orientation'] - df['mid3']
            df['mid4'] = df['orientation'] - df['mid4']
            df['theta1'] = df['orientation'] - df['theta1']
            df['theta2'] = df['orientation'] - df['theta2']
            df['theta3'] = df['orientation'] - df['theta3']
            df['theta4'] = df['orientation'] - df['theta4']
            
            print('All possible angles have been transformed. ')
            return df
        else:
            print('\nAngle type invalid. Alternatively, angle type cannot be transformed more than once.')
        continue_angle_manipulation = input('Continue with the angle manipulation: ')
        if i > 2:
            print('All possible angles have been transformed')
            
            return df
        if continue_angle_manipulation == "No":
            print('Very well, then. Angles will not be modified further.')
            
            return df
        i = i + 1
        
def drop_columns(df):
    all_dropped = 0
    print('Which columns do you want to drop? Type the column name which you wish to drop.\n You can also type "done", when you do not wish to drop anymore columns.')
    while all_dropped == 0:
        to_be_dropped = input('\nType the command here (type "options" to check all column options): ')
        if to_be_dropped == 'options':
            print('\nFollowing columns have been detected in the dataframe:', df.columns)
            print('\n')
            print('\nIn addition, you can drop multiple columns at the same time by typing following commands:')
            print('\nAlfa\nMid\nCurv\nCHL\nAreas\nArcs\ntheta\norientation')
            print('\n')
        elif to_be_dropped == 'Alfa':
            df = df.drop(['angle1','angle2','angle3','angle4'], 1)
        elif to_be_dropped == 'Mid':
            df = df.drop(['mid1','mid2','mid3','mid4'], 1)
        elif to_be_dropped == 'Curv':
            df = df.drop(['cu1','cu2','cu3','cu4'], 1)
        elif to_be_dropped == 'CHL':
            df = df.drop(['chL1','chL2','chL3','chL4'], 1)
        elif to_be_dropped == 'Areas':
            df = df.drop(['A1','A2','A3','A4'], 1)
        elif to_be_dropped == 'theta':
            df = df.drop(['theta1','theta2','theta3','theta4'], 1)
        elif to_be_dropped == 'Arcs':
            df = df.drop(['AL1','AL2','AL3','AL4'], 1)
        elif to_be_dropped == 'orientation':
            df = df.drop(['orientation'], 1)
        elif to_be_dropped == 'done':
            return df
        else:
            print('\nThere is no such command available. Check "options".')
            
def multiply(df):
    all_dropped = 0
    print('Which features do you want to sum up? Type the column name which you wish to drop.\n You can also type "done", when you do not wish to drop anymore columns.')
    while all_dropped == 0:
        mul_col = input('Type the feature here (check "options for commands"): ')
        if mul_col == 'options':
            print('\nYou can multiply columns at the same time by typing following commands:')
            print('\nAlfa\nMid\nCurv\nCHL\nAreas\nArcs\ntheta')
            print('\n')
        elif mul_col == 'Alfa':
            df['multiply_angle'] = df['angle1']*df['angle2']*df['angle3']*df['angle4']
        elif mul_col == 'Mid':
            df['multiply_mid'] = df['mid1']*df['mid2']*df['mid3']*df['mid4']
        elif mul_col == 'Curv':
            df['multiply_cu'] = df['cu1']*df['cu2']*df['cu3']*df['cu4']
        elif mul_col == 'CHL':
            df['multiply_chL'] = df['chL1']*df['chL2']*df['chL3']*df['chL4']
        elif mul_col == 'Areas':
            df['multiply_A'] = df['A1']*df['A2']*df['A3']*df['A4']
        elif mul_col == 'theta':
            df['multiply_theta'] = df['theta1']*df['theta2']*df['theta3']*df['theta4']
        elif mul_col == 'Arcs':
            df['multiply_arcL'] = df['AL1']*df['AL2']*df['AL3']*df['AL4']
        elif mul_col == 'done':
            return df
    
def add_sum(df):
    all_dropped = 0
    print('Which columns do you want to drop? Type the column name which you wish to drop.\n You can also type "done", when you do not wish to drop anymore columns.')
    while all_dropped == 0:
        mul_col = input('Type the feature here (check "options for commands"): ')
        if mul_col == 'options':
            print('\nYou can sum columns at the same time by typing following commands:')
            print('\nAlfa\nMid\nCurv\nCHL\nAreas\nArcs\ntheta')
            print('\n')
        elif mul_col == 'Alfa':
            df['sum_angle'] = df['angle1']+df['angle2']+df['angle3']+df['angle4']
        elif mul_col == 'Mid':
            df['sum_mid'] = df['mid1']+df['mid2']+df['mid3']+df['mid4']
        elif mul_col == 'Curv':
            df['sum_cu'] = df['cu1']+df['cu2']+df['cu3']+df['cu4']
        elif mul_col == 'CHL':
            df['sum_chL'] = df['chL1']+df['chL2']+df['chL3']+df['chL4']
        elif mul_col == 'Areas':
            df['sum_A'] = df['A1']+df['A2']+df['A3']+df['A4']
        elif mul_col == 'theta':
            df['sum_theta'] = df['theta1']+df['theta2']+df['theta3']+df['theta4']
        elif mul_col == 'Arcs':
            df['sum_arcL'] = df['AL1']+df['AL2']+df['AL3']+df['AL4']
        elif mul_col == 'done':
            return df
              
        
        
    

    
            
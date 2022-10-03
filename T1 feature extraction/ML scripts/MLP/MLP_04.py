'''
Created on 6 Jul 2020

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
import tsne_visualization
import feature_derivation
import pca_fitter
import math
from scipy import sparse

def main():
    datafile = input('Filename: ')
    print('\nKeep in mind that bad lines are skipped in original data format.')
    if datafile != 'test_plus': 
        df = pd.read_csv(datafile, sep="   ", header=None, error_bad_lines=False)
        print(df.shape)
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imp = imp.fit(df)
    #df = imp.transform(df)
        df = df.dropna()
    
    print('\nChosen filename: ',datafile)
    if datafile == "dataset_04.txt":
        df.columns = ['frame_number','T1_x','T1_y','1_mx','1_my','1_ex','1_ey','2_mx','2_my','2_ex','2_ey','3_mx','3_my','3_ex','3_ey','4_mx','4_my','4_ex','4_ey','orientation','label']
    elif datafile == 'test_1.txt':
        df.columns = ['frame_number', 'T1_x', 'T1_y','angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','orientation','label']
        #df = pd.DataFrame({'frame_number':df['frame_number'], 'T1_x': df['T1_x'], 'T1_y': df['T1_y'], 'angles': df['angle1']*df['angle2']*df['angle3']*df['angle4'], 'curvatures': df['cu1']*df['cu2']*df['cu3']*df['cu4']*pow(1000,4), 'chLs': df['chL1']*df['chL2']*df['chL3']*df['chL4'], 'thetas': df['theta1']*df['theta2']*df['theta3']*df['theta4'], 'ALs': df['AL1']*df['AL2']*df['AL3']*df['AL4'], 'orientation': df['orientation'], 'label': df['label']})
        print(df.head())
    elif datafile == 'test_plus':
        df1 = pd.read_csv('test_start_to_end_updated.txt', sep="   ", header=None, error_bad_lines=False)
        df2 = pd.read_csv('test_for_mid_point_updated.txt', sep="   ", header=None, error_bad_lines=False)
        df1 = df1.dropna()
        df2 = df2.dropna()
        df1.columns = ['frame_number', 'T1_x', 'T1_y','angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','A1','A2','A3','A4','orientation','label']
        df2.columns = ['frame_number', 'T1_x', 'T1_y','angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','A1','A2','A3','A4','orientation','label']
        df = df1
        df['mid1'] = df2['angle1']
        df['mid2'] = df2['angle2']
        df['mid3'] = df2['angle3']
        df['mid4'] = df2['angle4']
    elif datafile == 'test_10072020_normal.txt':
        df1 = pd.read_csv('test_10072020_normal.txt', delimiter="   ", header=None, error_bad_lines=False)
        df2 = pd.read_csv('test_10072020_curvature.txt', delimiter="   ", header=None, error_bad_lines=False)
        df1 = df1.dropna()
        df2 = df2.dropna()
        
        df1.columns = ['frame_number', 'T1_x', 'T1_y','angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','A1','A2','A3','A4','orientation','label']
        df2.columns = ['frame_number', 'T1_x', 'T1_y','angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','A1','A2','A3','A4','orientation','label']
        df = df1
        df['mid1'] = df2['angle1']
        df['mid2'] = df2['angle2']
        df['mid3'] = df2['angle3']
        df['mid4'] = df2['angle4']
        print(df['label'].mean()*100)
        
        
        df = df.drop(['cu1','cu2','cu3','cu4','theta1','theta2','theta3','theta4','mid1','mid2','mid3','mid4','orientation'],1)
        #df = pd.DataFrame({'frame_number':df['frame_number'], 'T1_x': df['T1_x'], 'T1_y': df['T1_y'], 'angles': df['angle1']*df['angle2']*df['angle3']*df['angle4'], 'chLs': df['chL1']*df['chL2']*df['chL3']*df['chL4'], 'ALs': df['AL1']*df['AL2']*df['AL3']*df['AL4'], 'mids': df['mid1']*df['mid2']*df['mid3']*df['mid4'], 'orientation': df['orientation'], 'label': df['label']})
        #print(df.head())
    elif datafile == 'test_start_to_end.txt' or 'test_for_mid_points.txt':
        df.columns = ['frame_number', 'T1_x', 'T1_y','angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','A1','A2','A3','A4','orientation','label']
        #df = pd.DataFrame({'frame_number':df['frame_number'], 'T1_x': df['T1_x'], 'T1_y': df['T1_y'], 'angles': df['angle1']*df['angle2']*df['angle3']*df['angle4'], 'curvatures': df['cu1']*df['cu2']*df['cu3']*df['cu4'], 'chLs': df['chL1']*df['chL2']*df['chL3']*df['chL4'], 'thetas': df['theta1']*df['theta2']*df['theta3']*df['theta4'], 'ALs': df['AL1']*df['AL2']*df['AL3']*df['AL4'], 'As': df['A1']*df['A2']*df['A3']*df['A4'], 'orientation': df['orientation'], 'label': df['label']})
        print(df.head())
    
    print('Dataset features: ',df.columns)
    print('--------------------------------------------------------------------------------------')
    print('\nData information section.')
    
    print('\nDataFrame description:')
    print(df.describe())
    print('\nDataFrame info:')
    print(df.info())
    print('--------------------------------------------------------------------------------------')
    data_manipulation = input('Use data manipulation? Type "Yes" or "No": ')
    if data_manipulation == "Yes":
        manipulation_type = input('\nWhich manipulation type do you want to use? Type answer here: ')
        if manipulation_type == "Angle-S1":
            print('Calculating direct angles between end points.')
            df = feature_derivation.angle(df)
        elif manipulation_type == "Angle-S2":
            print('Calculating direct angles between end points AND substracting them from flow orientation.')
            df = feature_derivation.angle(df)
            df = feature_derivation.orientation_strategy(df)
            
        else:
            print('\nDesired manipulation method has not yet been implemented. Exiting program.')
            exit()
    else:
        print('\nData will not be manipulated in any way.')
        print('--------------------------------------------------------------------------------------')
    strat = input('\nWhich stratagem do we use? 1: [50-50 + 50-50], 2: [50-50 + 10-90], 3: [10-90 + 10-90]: ')
    num_sample = input('\nWhat is the standard sample size? ')
    
    if int(strat) == 1:
        print('\nStratagem 1 chosen.')
        positives = df[df['label']==1]
        positives = positives.sample(n=int(int(num_sample)*0.5), random_state=1)
        zeros = df[df['label']==0]
        zeros= zeros.sample(n=int(int(num_sample)*0.5), random_state=1)
        frames = [positives, zeros]
        df = pd.concat(frames)
        labels = df['label']
        print(df.columns)
        inputs = df.drop(['frame_number','T1_x','T1_y', 'label'],1)
        input_train, input_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.20, random_state=0)
    elif int(strat) == 2:
        print('\nStratagem 2 chosen.')
        positives = df[df['label']==1]
        
        positives = positives.sample(n=int(int(num_sample)*0.5), random_state=1)
        print(positives.shape)
        zeros = df[df['label']==0]
        
        zeros_s= zeros.sample(n=int(int(num_sample)*0.5), random_state=1)
        print(zeros_s.shape)
        frames = [positives, zeros_s]
        df = pd.concat(frames)
        labels = df['label']
        
        inputs = df.drop(['frame_number','T1_x','T1_y','label'],1)
        input_train, input_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.20, random_state=0)
        positives_2 = positives.sample(n=int(int(num_sample)*0.1), random_state=1)
        print(positives_2.shape)
        #zeros_2 = df['label']
        #print(zeros_2.shape)
        zeros_2= zeros.sample(n=int(int(num_sample)*0.9), random_state=1, replace=False)
        print(zeros_2.shape)
        frames_2 = [positives_2, zeros_2]
        temp_df = pd.concat(frames_2)
        labels_2 = temp_df['label']
        
        inputs_2 = temp_df.drop(['frame_number','T1_x','T1_y','label'],1)
        input_train_2, input_test, labels_train_2, labels_test = train_test_split(inputs_2, labels_2, test_size=0.20, random_state=0)
        
    elif int(strat) == 3:
        print('\nStratagem 3 chosen.')
        positives = df[df['label']==1]
        positives = positives.sample(n=int(int(num_sample)*0.1), random_state=1)
        zeros = df[df['label']==0]
        zeros= zeros.sample(n=int(int(num_sample)*0.9), random_state=1)
        frames = [positives, zeros]
        df = pd.concat(frames)
        labels = df['label']
        
        inputs = df.drop(['frame_number','T1_x','T1_y','label'],1)
        input_train, input_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.20, random_state=0)
    else:
        print('\nInvalid stratagem. Exiting program.')
        exit()
    i = 0
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(input_train)
    input_train = imp.transform(input_train)
    input_test = imp.transform(input_test)
    solver_type = input('\nWhich ML solver do we use: ')
    print("\n------------------------------------------------------------")
    while i < 2:
        if i == 0:
            print('\nRunning the model with raw data (no scaling)...')
            print('Train set size:', input_train.shape)
        #9,10,11,8,
        if i != 0:
            do_pca = input('Do the PCA? Type "Yes" or "No": ')
            if do_pca == "Yes":
                pca = PCA(n_components=2, svd_solver='arpack', random_state=0)
                principalComponents = pca.fit_transform(input_train)
                #labels_2 = pd.concat(labels_train, labels_train)
                label_df = pd.DataFrame({'label': labels_train, 'label_2':labels_train})
                print('PCA',principalComponents.shape)
                print('labels',labels_train.shape)
                #print(labels_train.columns)
                finalDf =  np.hstack((principalComponents, label_df))
                #finalDf =  pd.concat(principalComponents, labels_train)
                #save_pca = input('\nSave PCA data? Type "Yes" or "No": ')
                #if save_pca == "Yes":
                #print(finalDf.head())
                #print(finalDf.transpose().head())
               
                print("\n------------------------------------------------------------")
                print("\nNumber of features in the original data: ",pca.n_features_)
                print(pd.DataFrame(pca.components_.transpose(),index=['angle1','angle2','angle3','angle4','cu1','cu2','cu3','cu4','chL1','chL2','chL3','chL4','theta1','theta2','theta3','theta4','AL1','AL2','AL3','AL4','A1','A2','A3','A4','orientation','mid1','mid2','mid3','mid4'],columns = ['PCA-1','PCA-2']))
                print("Data variance ratio after PCA tranform: ",pca.explained_variance_ratio_)
                print("\n------------------------------------------------------------")
                vis_pca = input('\nVisualize PCA components? Type "Yes" or "No": ')
                if vis_pca == "Yes":
                    fig = plt.figure(figsize = (8,8))
                    ax = fig.add_subplot(1,1,1) 
                    ax.set_xlabel('Principal Component 1', fontsize = 15)
                    ax.set_ylabel('Principal Component 2', fontsize = 15)
                    ax.set_title('PCA: 2 component projection', fontsize = 20)
                    targets = [0, 1]
                    colors = ['orangered', 'dodgerblue']
                    #print(finalDf.head())
                    for target, color in zip(targets,colors):
                        #finalDf_2 = finalDf[np.where(finalDf[:,2] == target)]
                        finalDf_2 = finalDf[finalDf[:,2] == target]
                        if color == 'r':
                            alfa = 1.0
                            normi = 1.0
                        else:
                            alfa = 0.80
                            normi=1.0
                        ax.scatter(finalDf_2[:,0], finalDf_2[:,1], c = color, alpha=alfa, marker='o', s=5.0, norm=normi)
                    ax.legend(targets)
                    ax.grid()
                    plt.show()
                    fit_pca = input('\nRun the MLP model with PCA fitted data? Type "Yes" or "No": ')
                    if fit_pca == "Yes":
                        pca_fitter.pca_model(pca, input_train, input_test, labels_train, labels_test, solver_type)
                    print('\nContinuing with the scaled fitting process..')
        clf = MLPClassifier(solver=solver_type, alpha=1e-5, hidden_layer_sizes=(12,8,6,), random_state=0, max_iter=20000, shuffle=True)   
        clf.fit(input_train, labels_train)
        predicted = clf.predict(input_test)
        conf_matrix = metrics.confusion_matrix(labels_test,predicted)
        tn, fp, fn, tp = metrics.confusion_matrix(labels_test, predicted).ravel()
        scores = cross_val_score(clf, input_test, labels_test, cv=5)
        
        print('Fitting of training data complete.')
        print('Predicting based on test data.')
        print("\nTraining set score: %f" % clf.score(input_train, labels_train))
        print("Test set score: %f" % clf.score(input_test, labels_test))
        print('Confusion matrix: \n')
        print('TN:', tn)
        print('TP: ',tp)
        print('FN: ',fn)
        print('FP:' ,fp)
        print('CROSS VALIDATION SCORES:', scores)
        print("\n------------------------------------------------------------")  
        if i == 0:
            print('\nScaling following features', inputs.columns)
            input_train = StandardScaler().fit_transform(input_train)
            input_test = StandardScaler().fit_transform(input_test)
            print('\nRunning the same model with scaled data...')
        i=i+1
    print('\nFinished.')
    
main()
    
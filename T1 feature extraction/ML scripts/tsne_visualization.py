'''
Created on 2 Jul 2020

@author: oskar, mehmet
'''
from sklearn.manifold import TSNE
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn import metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, average_precision_score
from sklearn.datasets import fetch_openml
import plots

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def foam_scatter(x, y):
    # choose a color palette with seaborn.
    
    finalDf =  np.hstack((x, y))
    print(finalDf.shape)
    print(finalDf[0])
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('x', fontsize = 15)
    ax.set_ylabel('y', fontsize = 15)
    ax.set_title('Data adjusted with t-SNE component vectors', fontsize = 20)
    targets = [0, 1]
    colors = ['orangered', 'dodgerblue']
    for target, color in zip(targets,colors):
        finalDf_2 = finalDf[np.where(finalDf[:,2] == target)]
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
    print('\nDone.')

def deliver_t_SNE(input_train, input_test, labels_train, labels_test):
    time_start = time.time()
    print('\nVery well, then. Starting t-SNE.')
    print('\nCalculating t-SNE...')
    foam_tsne = TSNE(random_state=0).fit_transform(input_train)
    print('\nt-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print('\nVisualizing t-SNE decomposition..')
    print('\nt-SNE shape:',foam_tsne.shape)
    print('Labels shape:',labels_train[:,0].reshape(-1,1).shape)
    #print('We get following estimation parameters: ',foam_tsne.get_params())
    
    
    foam_scatter(foam_tsne, labels_train[:,0].reshape(-1,1))
    save_data = input('Save t-SNE data to a separate file? Type "Yes" or "No": ')
    if save_data == "Yes":
        finalDf =  np.hstack((foam_tsne, labels_train))
       
    permission_to_proba = input('Visualize probability density map? Type "Yes" or "No": ')
    if permission_to_proba == "Yes":
        plots.initiate('t-SNE_data_04.txt')
    quick_predict = input('Finally, predict based on t-SNE data? Type "Yes" or "No": ')
    if quick_predict == "Yes":
        input_train, input_test, labels_train, labels_test = train_test_split(foam_tsne, labels_train, test_size=0.20, random_state=0)
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(14,20,8,4,), random_state=0, max_iter=20000, shuffle=True)   
        clf.fit(input_train, labels_train[:,0].reshape(-1,1).ravel())
        predicted = clf.predict(input_test)
        conf_matrix = metrics.confusion_matrix(labels_test[:,0].reshape(-1,1),predicted)
        tn, fp, fn, tp = metrics.confusion_matrix(labels_test[:,0].reshape(-1,1), predicted).ravel()
        print("\n------------------------------------------------------------")
        print('Fitting of training data complete.')
        print('Predicting based on test data.')
        print("\nTraining set score: %f" % clf.score(input_train, labels_train[:,0].reshape(-1,1)))
        print("Test set score: %f" % clf.score(input_test, labels_test[:,0].reshape(-1,1)))
        print('Precision score: ',precision_score(labels_test[:,0].reshape(-1,1), predicted, average='micro'))
        print('Recall score: ',recall_score(labels_test[:,0].reshape(-1,1), predicted, average='micro'))
        print('Confusion matrix: \n')
        print('TN:', tn)
        print('TP: ',tp)
        print('FN: ',fn)
        print('FP:' ,fp)
        print("\n------------------------------------------------------------")
    
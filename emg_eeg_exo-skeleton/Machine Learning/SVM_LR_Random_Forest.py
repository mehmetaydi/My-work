# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:58:15 2021

@author: mehmet
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


features_the_0904 = np.load('features_the_0904.npy')
features_mot_0904 = np.load('features_mot_0904.npy')


features_mot_0914 = np.load('features_mot_0914.npy')
features_the_0914 = np.load('features_the_0914.npy')


features_the_0903 = np.load('features_the_0903.npy')
features_mot_0903 = np.load('features_mot_0903.npy')


motion_plus = np.concatenate((features_mot_0904, features_mot_0914, features_mot_0903), axis=1)
therapy_plus = np.concatenate((features_the_0904, features_the_0914, features_the_0903), axis=1)

label_motion = np.ones((len(motion_plus[1, :, 1])))
label_therapy = np.zeros((len(therapy_plus[1, :, 1])))


final = np.concatenate((motion_plus, therapy_plus), axis=1)


label = np.hstack((label_motion, label_therapy))
xx = final.transpose(1, 0, 2)
x = xx.reshape((xx.shape[0], xx.shape[1] * xx.shape[2]))


X, y = shuffle(x, label, random_state=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)


# Import svm model

# Create a svm Classifier
clf = svm.SVC(kernel='rbf', gamma=1, C=1000, random_state=42)  # Linear Kernel


scores = cross_val_score(clf, X, y, cv=5)
print(scores)
# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy for SVM nomdel is :", metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


labels = ['0.0', '1.0']
confusion_matrix(y_test, y_pred, labels=[0, 1])


np.count_nonzero(y_test == 1)


# logistic regression


logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)

y_pred = logisticRegr.predict(X_test)

score = logisticRegr.score(X_test, y_test)
print("Accuracy for Logistic Regression nomdel is :", score)


# random forest

# Import the model we are using
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
# Train the model on training data
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

score = rf.score(X_test, y_test)
print("Accuracy for Randon forest nomdel is :", score)

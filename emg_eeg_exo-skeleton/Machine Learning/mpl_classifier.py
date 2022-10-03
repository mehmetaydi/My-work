# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:56:46 2021

@author: mehmet
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
# %matplotlib


features_the_0904 = np.load('features_the_0904.npy')
features_mot_0904 = np.load('features_mot_0904.npy')


features_mot_0914 = np.load('features_mot_0914.npy')
features_the_0914 = np.load('features_the_0914.npy')

features_freewalk_0914 = np.genfromtxt("features_freewalk_0914.txt", delimiter=',')


features_the_0903 = np.load('features_the_0903.npy')
features_mot_0903 = np.load('features_mot_0903.npy')

features_freewalk_0903 = np.genfromtxt("features_freewalk_0903.txt", delimiter=',')


motion_plus = np.concatenate((features_mot_0904, features_mot_0914, features_mot_0903), axis=1)
therapy_plus = np.concatenate((features_the_0904, features_the_0914, features_the_0903), axis=1)
free_walk = np.vstack((features_freewalk_0914, features_freewalk_0903))

label_motion = np.ones((len(motion_plus[1, :, 1])))
label_therapy = np.zeros((len(therapy_plus[1, :, 1])))


final = np.concatenate((motion_plus, therapy_plus), axis=1)


label = np.hstack((label_motion, label_therapy))
xx = final.transpose(1, 0, 2)
x = xx.reshape((xx.shape[0], xx.shape[1] * xx.shape[2]))

X, y = shuffle(x, label, random_state=0)

num_classes = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

y_train = to_categorical(y_train, num_classes)
y_testt = to_categorical(y_test, num_classes)

hidden_units = 256
dropout = 0.25

# Create the model
model = Sequential()
model.add(Dense(256, input_shape=(len(X_train[:, 1]), len(X_train[1, :])), activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(256, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation='softmax'))


opt = keras.optimizers.Adam(learning_rate=0.001)
# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_testt))


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
plt.figure()
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# Test the model after training
test_results = model.evaluate(X_test, y_testt, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

y_pred = model.predict_classes(X_test)

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print('AUC: %.3f' % auc)

plt.figure()
# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)

# Plot ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

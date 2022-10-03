# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 13:42:23 2021

@author: mehmet
"""


from os import listdir
from os.path import isfile, join
import numpy
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from keras.models import Sequential
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf
from matplotlib.pyplot import imshow
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import re
import warnings

# %matplotlib
warnings.filterwarnings("ignore")


def extract_number(string):
    r = re.compile(r'(\d+)')
    return int(r.findall(string)[0])


dim = (300, 300)
mypath = r'C:\Users\phmeay\Desktop\Work files\DTF-PDC\OPDC\OPDC\image files'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
sortedFiles = sorted(onlyfiles, key=lambda x: extract_number(x))
images = np.empty((len(sortedFiles), 300, 300))
for n in range(0, len(sortedFiles)):
    img = cv2.imread(join(mypath, sortedFiles[n]), cv2.IMREAD_GRAYSCALE)
    images[n] = cv2.resize(img, dim)

# np.save('images.npy',images)

label_motion = np.ones((len(images[:, 1, 1]))-952)
label_therapy = np.zeros((len(images[:, 1, 1]))-663)
# label_free_walk = 2*np.ones((len(images[:,1,1]))-1615)


x = images.reshape((images.shape[0], 300, 300, 1))
label = np.hstack((label_motion, label_therapy))

# np.savetxt('label.txt',label)

X, y = shuffle(x, label, random_state=0)


num_classes = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)


# y_testt =y_test
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# convert from integers to floats
train_norm = X_train.astype('float32')
test_norm = X_test.astype('float32')
# normalize to range 0-1
X_train = train_norm / 255.0
X_test = test_norm / 255.0


################ using K-Fold #########################
acc_per_fold = []
loss_per_fold = []

inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Model configuration

batch_size = 50
no_epochs = 50

verbosity = 1
num_folds = 5
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

    # Define the model architecture
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(64, 64, 1)),
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


#####################################################################################################################


################# No K-fold #####################
y_testt = y_test
y_train = to_categorical(y_train, num_classes)
y_testt = to_categorical(y_test, num_classes)


model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(300, 300, 1)),
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])


opt = keras.optimizers.Adam(learning_rate=0.001)
# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, verbose=1, validation_data=(X_test, y_testt))


# Get training and validation loss histories
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


# Get training and validation accuracy histories
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Create count of the number of epochs
plt.figure()
epoch_count = range(1, len(training_acc) + 1)

# Visualize accuracy history
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, val_acc, 'b-')
plt.legend(['Training accuracy', 'Test accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# Test the model after training
test_results = model.evaluate(X_test, y_testt, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

y_pred = model.predict_classes(X_test)

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print(classification_report(y_test, y_pred))

# roc = {label: [] for label in np.unique(label)}
# for label in np.unique(label):
#     model.fit(X_train, y_train == label)
#     predictions_proba = model.predict_proba(y_test)
#     roc[label] += roc_auc_score(y_test, predictions_proba[:,1])
# print('AUC: %.3f' % roc)

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

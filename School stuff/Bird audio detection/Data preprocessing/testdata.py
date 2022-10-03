# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:13:04 2020

@author: mehmet
"""



import os
import numpy
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import librosa

import numpy as np

import pandas as pd
import glob
import sys
my_path = r"C:\Users\mehmet\Desktop\test\\data"
epsulon = sys.float_info.epsilon
audio_name =[]
for dir2 in os.listdir(my_path):
    for file1 in os.listdir(os.path.join(my_path, dir2)):
        
        file_path= os.path.join(my_path, dir2,  file1)

        audio_name.append(file_path)

import natsort 
b = (natsort.natsorted(audio_name,reverse=False))
dt =[]
X_sort =[]
for i in range(0,len(b)):
    data_set2 = np.load(b[i])
    # data_set2, samplerate= librosa.load(b[i], sr=None)
    samplerate = len(data_set2)/10
    data_set2 = data_set2 * 1/np.max(np.abs(data_set2))
    
    dt.append(data_set2)
    
    data2 =librosa.resample(data_set2, samplerate, 44100)
      
    melspectrogram_test2 = librosa.feature.melspectrogram(y=data2,sr=samplerate,n_fft=1024, hop_length=512, n_mels=40 ,power =1)
    melspectrogram1 = 20*np.log10(melspectrogram_test2+epsulon)
    # melspectrogram1 = melspectrogram1 / np.max(np.abs(melspectrogram1))
    X_sort.append(melspectrogram1)
    
dt = np.array((dt)) 
X_sort = np.array(X_sort)   
numpy.save("data.npy",X_sort)

# kk = np.load('7200.npy')

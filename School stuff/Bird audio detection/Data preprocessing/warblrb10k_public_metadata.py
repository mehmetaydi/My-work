# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:39:48 2020

@author: mehmet
"""
import os
import numpy
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import librosa
import numpy as np
import pandas as pd
import glob
import sys

test_folder=r'C:\Users\mehmet\Desktop\test\small data\wav'
def test_dataset_PIL(test_folder):
    
    audio_name = []
    audio_data =[]
    for aud in os.listdir(test_folder):
        
        audio_namee = os.path.basename(aud)
        audio_name.append(audio_namee)
        
        audio_path= os.path.join(test_folder, aud)
        data_train, samplerate= librosa.load(audio_path, sr=None)
        data_train = data_train * 1/np.max(np.abs(data_train))
        # data_train1 =librosa.resample(data_train, samplerate, 44100)
        data_train = data_train[:441000,]
        data_train = np.resize(data_train, 441000)
       
        
        audio_data.append(data_train)
        
    return audio_data, audio_name ,samplerate


data, audio_name ,samplerate= test_dataset_PIL(test_folder)
data = np.asarray(data)
names = pd.DataFrame(audio_name,columns=['itemid'])
names['itemid'] = names.itemid.str[:-4] 
label = pd.read_csv(r'C:\Users\mehmet\Desktop\test\warblrb10k_public_metadata_2018.csv', delimiter=',')
epsilon = sys.float_info.epsilon


y = pd.merge(names, label, on='itemid')

y = y['hasbird']
y =y.to_numpy()

np.savetxt('8000.txt', y, delimiter=',') 
 = []
mfcc_train =[]
for i in range(0,len(data)):
    melspectrogram_train = librosa.feature.melspectrogram(y=data[i],sr=samplerate,n_fft=1024, hop_length=512, n_mels=40 ,power =1)
    melspectrogram = 20*np.log10(melspectrogram_train+epsilon)
    mfcc1 = librosa.feature.mfcc(y=data[i], sr=samplerate, S=None, n_mfcc=40)
    mfcc_train.append(mfcc1)
    # melspectrogram = melspectrogram / np.max(np.abs(melspectrogram))
    X_train.append(melspectrogram)
X  =np.array(X_train)
mfcc_train =np.array(mfcc_train)
# numpy.save("8000.npy",X)

librosa.display.specshow(X_train[5], x_axis='time')
# librosa.display.specshow(mfcc_train[1,:,:], x_axis='time')


# plt.plot(mfcc_train[0,:,:])

# plt.plot(mfcc_train[1,:,:])
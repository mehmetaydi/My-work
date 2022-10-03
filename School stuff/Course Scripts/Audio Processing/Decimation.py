# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:47:11 2020

@author: mehmet
"""
import scipy.io.wavfile
import scipy.signal 
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq
from scipy import ndimage
import numpy as np
# in k the first column refer to frequency 2. one is amplitude 3. on is phase
k = np.array([[100, 1,0.5], [500, 1.3,0.8], [1500, 1.5,0.25], [2500, 2,0.9]])
f_s = 8000  # Sampling rate, or number of measurements per second
x_total = []
t = np.linspace(0, 3,  3*f_s, endpoint=False)
for i in range(0,len(k)):          
    x = k[i,1]*np.sin(k[i,0] * 2 * np.pi * t+ k[i,2])
   
    x_total.append(x)    

x_total =np.array(x_total).transpose()

######### summation of the created sinusoids ############
sum_of_sinusoids = []    
for m in range(0,len(x_total)):
    added_sinusoid = x_total[m,0] + x_total[m,1]+ x_total[m,2]+ x_total[m,3]
    sum_of_sinusoids.append(added_sinusoid)
sum_of_sinusoids = np.array(sum_of_sinusoids) 

# downsampling_sinusoid = scipy.signal.resample(sum_of_sinusoids, 12000, t=None, axis=0)# resemblig data by factor 2  
downsampling_sinusoid = ndimage.interpolation.zoom(sum_of_sinusoids,0.5) # resemblig data by factor 2 
scipy.io.wavfile.write('downsampling_sinusoid.wav',f_s, downsampling_sinusoid)
nfft = len(downsampling_sinusoid)
fig, ax = plt.subplots()
ax.plot(downsampling_sinusoid)
y = fft(downsampling_sinusoid,nfft)
y = np.abs(y)
fig, ax = plt.subplots()
fs = f_s/2
fre = np.arange(nfft) * (fs/nfft)
# fre1 = np.arange(f_s)
# ax.plot(len(x),y1)# missing value
ax.plot(fre[:len(fre)//2+1],y[:len(fre)//2+1])
# ax.plot(y)
plt.title('DFT of downsampled sinusoids')
fig, ax = plt.subplots()
ax.plot(sum_of_sinusoids)
y1 = fft(sum_of_sinusoids)
y1= np.abs(y1)
fig, ax = plt.subplots()

# ax.plot(y1)
fre1 = np.arange(nfft) * (f_s/nfft)
ax.plot(fre1[:len(fre1)//2+1],y1[:len(fre1)//2+1])
plt.title('DFT of summed sinusoids')
plt.show()
    
    
    

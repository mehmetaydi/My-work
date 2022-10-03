# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:24:10 2021

@author: mehmet
"""

import matplotlib.pyplot as plt
import pywt
import numpy as np
import neurokit2 as nk
%matplotlib

# Get data:
data = np.genfromtxt(r"emg_threapy.txt",delimiter=",", dtype =float) # threapy plus
# data = np.genfromtxt(r"emg_motion.txt",delimiter=",", dtype =float) # motion plus
# data = np.genfromtxt(r"emg_motion_no_norm.txt",delimiter=",", dtype =float) # motion plus
file_object  = open('./ecg_emg.txt', 'r')
raw_data = file_object.readlines()
data = data-np.mean(data)

# Create wavelet object and define parameters
w = pywt.Wavelet('db4')


maxlev = pywt.dwt_max_level(len(data), w.dec_len)
# maxlev = 2 # Override if desired
print("maximum level is " + str(maxlev))
# threshold =0.01# Threshold for filtering

# threshold =np.arange(0.01, 0.02, 0.01)
# for threshold in threshold:
    
    
# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'db4', level=maxlev)

#cA = pywt.threshold(cA, threshold*max(cA))
# plt.figure()
# for i in range(1, len(coeffs)):
# plt.subplot(maxlev, 1, i)
# plt.plot(coeffs[i])
coeffs[10] = pywt.threshold(coeffs[10], 0.005*np.std(coeffs[10]),'less')
coeffs[11] = pywt.threshold(coeffs[11], 0.005*np.std(coeffs[11]),'less')
coeffs[12] = pywt.threshold(coeffs[12], 0.002*np.std(coeffs[12]),'less')
# coeffs[i] = pywt.threshold_firm(coeffs[i], threshold*np.std(coeffs[i]),2)


datarec = pywt.waverec(coeffs, 'db4')



_, rpeaks = nk.ecg_peaks(data, sampling_rate=100)
# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks['ECG_R_Peaks'], data)

# Zooming into the first 5 R-peaks
# plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], data[:20000])
peak_points = rpeaks['ECG_R_Peaks']

for i in peak_points:
    data[(i-5):(i+5)] = data[i+5]

data_new = data
# data = np.genfromtxt(r"emg_motion.txt",delimiter=",", dtype =float) # motion plus
data = np.genfromtxt(r"emg_threapy.txt",delimiter=",", dtype =float) # threapy plus
# plt.figure()
fig, ax = plt.subplots(figsize=(20,10)) 
plt.subplot(3, 1, 1)
plt.plot( data)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(3, 1, 2)
plt.plot(datarec)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
# plt.ylim(-0.01,0.03)
plt.title("De-noised signal using wavelet transform")
# plt.suptitle("Threshold {}".format(threshold))

plt.subplot(3, 1, 3)
plt.plot(data_new)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
# plt.ylim(-0.01,0.03)
plt.title("De-noised signal using R peak model")

plt.tight_layout()
plt.show()






























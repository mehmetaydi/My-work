# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:04:22 2021

@author: phmeay
"""

# import cv2
from PIL import Image
import numpy as np
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import librosa
import librosa.display
%matplotlib


fs = 100
easy_file = np.genfromtxt("20210414143751_Patient01.easy")
easy_file = easy_file[9776:, :]

left_leg_walk_points = np.genfromtxt("left_leg.txt")
right_leg_walk_points = np.genfromtxt("right_leg.txt")

legs = np.hstack((left_leg_walk_points, right_leg_walk_points))


def filtering(m):
    xc = signal.resample_poly(m, 1, 5, padtype='mean')
    fs = 100
    k = xc / max(abs(xc))

    b1, a1 = signal.butter(4, [0.1/(fs/2), 30/(fs/2)], 'bandpass')
    Signal = signal.filtfilt(b1, a1, k)
    return Signal


si = filtering(easy_file[:, 1])

filtered_EASY_FILE = np.zeros((20, len(si)))

for i in range(len(easy_file[1, :])-5):
    sig = easy_file[:, i]

    filtered_EASY_FILE[i, :] = filtering(sig)


#####  Event related potential for channel C1 ####

C1 = filtered_EASY_FILE[0, :]

signal = C1[5927:6077]


def calculate_statistics(x):
    features_last = []
    for ii in range(len(erp[:, 1])):
        coeffs = pywt.wavedec(x[ii, :], 'db4', level=4)
        cA4, cD4, cD3, cD2, cD1 = coeffs
        new_feat = []

        for list_values in coeffs:
            max_val = np.max(list_values)
            min_val = np.min(list_values)
            median = np.nanpercentile(list_values, 50)
            mean = np.nanmean(list_values)
            std = np.nanstd(list_values)
            var = np.nanvar(list_values)
            rms = np.nanmean(np.sqrt(list_values**2))
            feature = np.vstack((max_val, min_val, median, mean, std, var, rms))
            new_feat.append(feature)
        features = np.ravel(new_feat)
        features_last.append(features)
    return features_last


channels_order = ['Pz', 'O1', 'O2', 'C4', 'C2', 'Fz', 'C3', 'C1', 'Cz']

channels_rest = np.vstack((filtered_EASY_FILE[3, :], filtered_EASY_FILE[6, :], filtered_EASY_FILE[7, :], filtered_EASY_FILE[10, :], filtered_EASY_FILE[5, :], filtered_EASY_FILE[13, :],
                           filtered_EASY_FILE[14, :], filtered_EASY_FILE[0, :], filtered_EASY_FILE[3, :]))

channels_0904 = np.vstack((filtered_EASY_FILE[1, :], filtered_EASY_FILE[2, :], filtered_EASY_FILE[3, :], filtered_EASY_FILE[10, :], filtered_EASY_FILE[12, :], filtered_EASY_FILE[13, :],
                           filtered_EASY_FILE[14, :], filtered_EASY_FILE[15, :], filtered_EASY_FILE[18, :]))


all_features = []
for jj in range(len(channels_rest)):
    erp = []
    reversed_signal = []
    moving_point = np.zeros((len(legs)))
    for ind, j in enumerate(legs):
        segment = channels_rest[jj, :][int(j-0.5*fs):int(j+fs)]
        reversed_signall = segment[::-1]
        reversed_signal.append(reversed_signall)
        erp.append(segment)
        moving_point[ind] = segment[49]

    reversed_signal = np.array(reversed_signal)
    erp = np.array(erp)

    features1 = calculate_statistics(erp)
    features1 = np.array(features1)

    an_array = np.column_stack((features1, moving_point))
    all_features.append(an_array)

all_features = np.array(all_features)


np.save('features_freewalk_0914', all_features)

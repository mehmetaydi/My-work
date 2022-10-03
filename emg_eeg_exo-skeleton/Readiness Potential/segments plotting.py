

# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:57:45 2021

@author: mehmet
"""
import numpy as np
import cv2
from scipy import signal
import matplotlib
import  pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz
import mne
from PIL import Image
from scipy import ndimage
import mplcursors
import scipy.signal
import os
import librosa
from scipy import signal
import shutil
import warnings
%matplotlib 
fs =500 # sampling frequency of EEC signal
warnings.filterwarnings("ignore")  

date = '09.04.2021'

def files104849():
    easy_file = np.genfromtxt("20210409104849_Patient01.easy")
    firstEEGtimestamp = easy_file[0:1,24]
    edf_file = "20210409104849_Patient01.edf"
    easy_file = np.transpose(easy_file)
    # reading edf file     
    data = mne.io.read_raw_edf(edf_file)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    return easy_file , firstEEGtimestamp,info

def files102517():
    easy_file = np.genfromtxt("20210409102517_Patient01.easy")
    firstEEGtimestamp = easy_file[0:1,24]
    edf_file = "20210409102517_Patient01.edf"
    easy_file = np.transpose(easy_file)
    # reading edf file     
    data = mne.io.read_raw_edf(edf_file)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    return easy_file , firstEEGtimestamp,info
    
    
  

xx = int(input('Enter data frame that you want to execute \n press 1 if you want to visualize files102517 ( Motion plus)  \n press 2 if you want to visualize files104849 (Therapy plus ) : '))

if xx == 1:
    file_name = "102517_Motion plus"
    # left leg
    text_data_resempled_left = np.genfromtxt('210409102503_left_edited_100Hz.txt', delimiter=',') # Resampled data for left leg  from matlab y = resample(x,tx,fs) fs taken 100 Hz
    easy_file_left , firstEEGtimestamp_left,info = files102517()
    easy_file_left = easy_file_left[:,278:]
    firstEEGtimestamp_left = easy_file_left[24,0]
    # right leg
    text_data_resempled_right = np.genfromtxt('210409102508_right_edited_100Hz.txt', delimiter=',')
    easy_file_right , firstEEGtimestamp_right,info = files102517()
    easy_file_right = easy_file_right[:,278:]
    firstEEGtimestamp_right = easy_file_right[24,0]
    
   
if xx == 2:
    file_name = "104849_Therapy plus"
    # left leg
    text_data_resempled_left = np.genfromtxt('210409104837_left_edited_100Hz.txt', delimiter=',')
    easy_file_left , firstEEGtimestamp_left,info = files104849()
    easy_file_left = easy_file_left[:,209:]
    firstEEGtimestamp_left = easy_file_left[24,0]
    # right leg
    text_data_resempled_right = np.genfromtxt('210409104842_right_edited_100Hz.txt', delimiter=',')
    easy_file_right , firstEEGtimestamp_right,info= files104849()
    easy_file_right = easy_file_right[:,209:]
    firstEEGtimestamp_right = easy_file_right[24,0]
    

   
if xx not in range(1,3):
    raise ValueError("Please enter a valid number")


directory = "{}".format(file_name)
parent_dir = input("Please specify the directory where the figures be stored:")
parent_dir =r"{}".format(parent_dir)

path = os.path.join(parent_dir, directory)
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)



EEG_signal = easy_file_right[0:19,:]


ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])   

EEG_signals = np.array([np.vstack((EEG_signal[16],EEG_signal[11])),np.vstack((EEG_signal[0],EEG_signal[13],EEG_signal[9])),np.vstack((EEG_signal[17],EEG_signal[8])),
                        np.vstack((EEG_signal[14],EEG_signal[15],EEG_signal[18],EEG_signal[12],EEG_signal[10])), np.vstack((EEG_signal[7],EEG_signal[1],EEG_signal[6])), np.vstack((EEG_signal[2],EEG_signal[3]))])

EEG_signal_name1 =[['AF3','AF4'],['F7','Fz','F8'],['FC5','FC6'],['C3','C1','Cz','C2','C4'],['P7','Pz','P8'],['O1','O2']]

signal_channels = ["Z_channels","C_channels","odd_channels","even_channels","AF channels","FC channels"]



signal_left_all_together =[]
signal_right_all_together =[]
for EEG_signals,EEG_signal_name, signal_channels in zip(EEG_signals,EEG_signal_name1, signal_channels):
    def EEG_Signal(EEG_signal_name,EEG_signals,signal_channels):
        EEG_signal_name =EEG_signal_name
        EEG_signals = EEG_signals
        signal_channels =signal_channels
        return EEG_signal_name, EEG_signals,signal_channels
    
  
    txt_data = np.array([text_data_resempled_left,text_data_resempled_right])
    easy_file =np.array([easy_file_left,easy_file_right])
    firstEEGtimestamp =np.array([firstEEGtimestamp_left,firstEEGtimestamp_right])


    EEG_signal_name, EEG_signals,signal_channels = EEG_Signal(EEG_signal_name,EEG_signals,signal_channels)
    
    directory2 = "{}".format(signal_channels)
   
    parent_dir =r"{}".format(parent_dir)
    
    mypath = os.path.join(parent_dir, directory,directory2)
    if os.path.exists(mypath):
        shutil.rmtree(mypath)
    os.makedirs(mypath)
    
    
    def filtering(m):
        xc = signal.resample_poly(m, 1, 5, padtype='mean')
        fs = 100
        k = xc / max(abs(xc)) 
         
        b1, a1 = signal.butter(4, [0.1/(fs/2),30/(fs/2)], 'bandpass')
        Signal = signal.filtfilt(b1, a1, k)
        return Signal
    
    movement =[]
    c_ler =[]
    w_z =[]
    original_accelerometer =[]
    right_leg_movement=[]
    left_leg_movement=[]
    
    for index,( easy_file,firstEEGtimestamp,txt_data) in enumerate(zip(easy_file,firstEEGtimestamp,txt_data)):
        
        
        xx = xx
        firstEEGtimestamp =int(firstEEGtimestamp)
    
        wz = txt_data
    
        
        def crossings_zero_pos2neg(data):
            pos = data > 0
            return (pos[:-1] & ~pos[1:]).nonzero()[0]
        
        
        unixtime_in_easyfile =easy_file[24]
    
        wzz = np.where( (wz > -40) & (wz <53), 0, wz) 
        
        wzz = np.where(wzz==0, 0.001, wzz)
        c = crossings_zero_pos2neg(wzz)
        original_accelerometer.append(wz)
        c_ler.append(c)
        w_z.append(wzz)
        if xx == 1:
            if index ==0:
                c_left = c.copy()
                segment = [c_left[0:8],c_left[8:16],c_left[16:26],c_left[26:35],c_left[35:46],c_left[46:56],c_left[56:71],c_left[71:84],c_left[85:90]]
                left_leg_movement.append(segment)
    
                
            if index ==1:
                c_right = c.copy()
                segment = [c_right[0:8],c_right[8:16],c_right[16:26],c_right[26:36],c_right[36:48],c_right[48:59],c_right[59:74],c_right[74:87],c_right[88:95]]
                right_leg_movement.append(segment)
    
        if xx == 2:
            
            if index ==0:
                c_left = c.copy()
                segment = [c_left[0:9],c_left[9:20],c_left[20:31],c_left[31:42],c_left[42:51],c_left[51:60],c_left[60:70],c_left[70:80],c_left[80:88],c_left[88:,]]
                left_leg_movement.append(segment)
    
            if index == 1:   
                c_right = c.copy()
                segment = [c_right[0:10],c_right[10:22],c_right[22:33],c_right[33:45],c_right[45:54],c_right[54:64],c_right[64:74],c_right[74:85],c_right[85:96],c_right[96:,]]
                right_leg_movement.append(segment)
    
                
        eeg_signal_name, eeg_signal, signal_channels = EEG_Signal(EEG_signal_name,EEG_signals,signal_channels)
            
        for m,j in zip(eeg_signal, eeg_signal_name):
    
            move =[]
    
            movement.append(move)
            for idx, p in enumerate(segment):
    
                Signal = filtering(m)
                movement_time_in_edf=[]
                move.append(movement_time_in_edf)
                for i in range(len(p)):
                    starting_point = p[i]
                    corresponding_time_stamp = unixtime_in_easyfile[starting_point]
    
                    firstEEGtimestamp = firstEEGtimestamp                
                    movement_time_in_edf1 = int(abs(firstEEGtimestamp-corresponding_time_stamp)/(1000/fs))
                    movement_time_in_edf.append(movement_time_in_edf1)
                
    
    def repeated_variables(t): # This def refers to repeated variables in the other functions below
        fs =100
        movement_left = movement1[z][t]
        movement_right = movement2[z][t]
        min_point = min(np.concatenate((movement_left, movement_right)))
        max_point = max(np.concatenate((movement_left, movement_right)))
        position_label = np.arange(np.ceil((min_point)/fs), np.ceil((max_point)/fs), 5)   
        signal_for_each_segment=Signal[min_point:max_point]
        labels = np.around(position_label, decimals=2) 
        positions = np.arange(min_point, max_point, 5*fs)
        return movement_left,movement_right,min_point,max_point,position_label,signal_for_each_segment,labels,positions 
        
    

    
    def averaging_over_walks(before, after): 
        klm =0        
        signal_right =[]
        signal_left =[]
        
        for em,tte in zip( Signal_all, eeg_signal_name):
            fig, axes = plt.subplots(nrows=int(len(segment)/2), ncols=2,figsize=(20,10),sharey=True) 
            axes = axes.ravel() 
            walk_number =np.tile(np.arange(0,int(len(segment)/2)),2) 
            if (len(segment) % 2) == 0:
                ind =list(sorted(np.arange(0,int(len(segment))), key=lambda x: [x % 2, x])) # index of the plot odd number refers to left leg even numbers are right leg
            else:
                ind =list(sorted(np.arange(0,int(len(segment))-1), key=lambda x: [x % 2, x]))
            walk_name =['Left Leg','Right Leg']      
            ct = 0
            
            Sig_right =[]
            signal_right.append(Sig_right)
            klm = klm+1
            Sig_left =[]
            signal_left.append(Sig_left)
            for t in range(int(len(segment)/2)):
                fs =100
                count =1
                
                # globals()['left_leg_walk'+str(t+1)] = np.hstack((left_leg_movement[0][ct],left_leg_movement[0][ct+1]))
                # globals()['right_leg_walk'+str(t+1)] = np.hstack((right_leg_movement[0][ct],right_leg_movement[0][ct+1]))
                left_leg_walk= np.hstack((left_leg_movement[0][ct],left_leg_movement[0][ct+1]))
                right_leg_walk= np.hstack((right_leg_movement[0][ct],right_leg_movement[0][ct+1]))
                ct=ct+2
                sig_left =[]
                
                sig_right =[]
                
                for left_leg_walk, right_leg_walk in zip(left_leg_walk, right_leg_walk):
                    sig_leftt = em[int(left_leg_walk-fs*before):int(left_leg_walk+fs*after)]
                    sig_rightt = em[int(right_leg_walk-fs*before):int(right_leg_walk+fs*after)]
                    sig_left.append(sig_leftt)
                    sig_right.append(sig_rightt)
              
                sig_left =np.array(sig_left)
                sig_left =sig_left.mean(axis=0)
                Sig_left.append(sig_left) 
                sig_right =np.array(sig_right)
                sig_right = sig_right.mean(axis=0)
                Sig_right.append(sig_right)
            avr1 = Sig_left + Sig_right 
            avr =np.array(avr1)
            max_value,min_value =np.max(avr),np.min(avr)
            count = 0
            for i, avr, walk_number in (zip(ind,avr,walk_number)): 
                axes[i].plot(avr,linewidth=0.5)
                axes[i].axvline(x=(before*len(sig_leftt)/(before+after)),linewidth=0.4, c ='r', linestyle="--")
                plt.suptitle(' {} {} signal'.format(file_name,tte), y=1.00)
                axes[count].set(xlabel='Samples', ylabel='Amplitude') 
                if i in [0,1]:
                    axes.flat[i].set_title('{} '.format(walk_name[i])+'\n\n{} walk'.format(ordinal(walk_number+1)))
                else:
                    
                    axes.flat[i].set_title('{} walk'.format(ordinal(walk_number+1))) 
                axes[count].set_ylim([min_value,max_value ])
                count =count+1
                plt.tight_layout() 
            plt.savefig(mypath + "/{} Signal.pdf".format(tte)) 
        return signal_left,signal_right,walk_name
    
    def plot_each_segment_of_signal():
                
        for t in range(len(segment)):
            fs =100
            count =1
            fig, axes = plt.subplots(nrows=len(eeg_signal)+1, ncols=1,figsize=(20,10)) 
            for z,em,tte in zip(interval, Signal_all, eeg_signal_name):
                
                movement_left,movement_right,min_point,max_point,position_label,signal_for_each_segment,labels,positions = repeated_variables(t)
                signal_for_each_segment1=em[np.min(movement_left):np.max(movement_left)]
                axes[0].plot(original_accelerometer[0], linewidth=0.5,c = 'r')
                axes[0].set_xlim([min_point, max_point])
                axes[0].plot(original_accelerometer[1], linewidth=0.5,c = 'g')
                axes[0].set_xlim([min_point-fs, max_point+fs])
                axes[0].set(xlabel='Sample', ylabel='Amplitude')
                axes[0].set_xticks(positions)
                axes[0].set_xticklabels(labels)
                for xc in right_leg_movement[0][t]:
                    xc =np.array(xc)
                    axes[0].axvline(x=(xc), linewidth=0.4,c = 'g')
                for xc in left_leg_movement[0][t]:
                    xc =np.array(xc)
                    axes[0].axvline(x=(xc), linewidth=0.4,c = 'r')        
                axes[count].plot(em, linewidth=0.5)
                axes[count].set_title('{} signal {} part'.format(tte,ordinal(t+1)))           
                axes[count].set(xlabel='Time (sec)', ylabel='Amplitude')            
                axes[count].set_xticks(positions)
                axes[count].set_xticklabels(labels)
                axes[count].set_xlim([min_point-fs, max_point+fs])
                axes[count].set_ylim([np.min(signal_for_each_segment1), np.max(signal_for_each_segment1)])
                mplcursors.cursor()
                for xc in movement_left:
                    axes[count].axvline(x=xc, linewidth=0.4,c ='r')
                for pkk in movement_right:
                    axes[count].axvline(x=pkk,linewidth=0.4, c ='g')           
                plt.suptitle(' {} '.format(file_name), y=1.00)
                           
                count+=1        
                plt.tight_layout() 
                plt.savefig(mypath + "/{} {} segment.pdf".format(signal_channels,ordinal(t+1))) 
            
                            
    def plot_segment_by_segment():
        fs =100
        
        for t in range(len(segment)):
            fig, ax = plt.subplots(figsize=(20,10))         
            movement_left,movement_right,position_label,signal_for_each_segment = repeated_variables(t)
            plt.plot(Signal)
            ax.set_title('{} signal {} part'.format(tt,ordinal(t+1)))
            plt.xlabel('Second (sec)',loc='center')
            plt.ylabel('Amplitude (uV)',loc='center')
            plt.xlim([np.min(movement_left)-3*fs, np.max(movement_left)+fs])
            plt.ylim([np.min(signal_for_each_segment), np.max(signal_for_each_segment)])
            labels = np.around(position_label, decimals=2)
            positions = np.arange(np.ceil(np.min(movement_left)-3*fs), np.ceil(np.max(movement_left)+fs), 5*fs)
            plt.xticks(positions, labels) 
            mplcursors.cursor()
            for xc in movement_left:
                plt.axvline(x=xc, c ='r')
            for pkk in movement_right:
                plt.axvline(x=pkk, c ='g')
            plt.show()  
         
    
    def plot_all():
        fs =100
        if xx == 1 or xx==2:
            fig = plt.figure(figsize=(20,10)) 
            gs = gridspec.GridSpec(6, 2)
        if xx==3:
            fig= plt.figure(figsize=(20,10)) 
            gs = gridspec.GridSpec(4, 2)
        for t in range(len(segment)):
            
            movement_left,movement_right,min_point,max_point,position_label,signal_for_each_segment,labels,positions = repeated_variables(t)
            if t % 2 == 0:
                x_axis = int(((t+1)/2)+1)
                p =0
            else: 
                p =1
                x_axis = int((t+1)/2)
            ax = fig.add_subplot(gs[0, :]) 
            
            ax.plot(original_accelerometer[0], linewidth=0.5,c ='r')
            ax.plot(original_accelerometer[1], linewidth=0.5,c ='g')
            ax.set(xlabel='Samples', ylabel='Angular Vel')
            for xc in c_left:
                ax.axvline(x=xc, linewidth=0.4, c ='r')
            ax.set(xlabel='Samples', ylabel='Angular Vel')
            for xc in c_right:
                ax.axvline(x=xc, linewidth=0.4, c ='g')
            ax = fig.add_subplot(gs[x_axis,p]) 
            def plot_figure(ax):
                
                ax.plot(Signal,linewidth=0.5)
                ax.set_title('{} signal {} part'.format(tt,ordinal(t+1)))      
                ax.set(xlabel='Time (sec)', ylabel='Amplitude')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)
                ax.set_xlim([min_point-3*fs, max_point+fs])
                ax.set_ylim([np.min(signal_for_each_segment), np.max(signal_for_each_segment)])
                mplcursors.cursor()
                for xc in movement_left:
                    ax.axvline(x=xc, linewidth=0.4,c ='r')
                for pkk in movement_right:
                    ax.axvline(x=pkk,linewidth=0.4, c ='g') 
   
            if xx ==3 or xx==1:
                if t == len(segment) - 1:
                    c =int(len(segment)/2+0.5)
                    ax = fig.add_subplot(gs[c, :]) 
                    plot_figure(ax)
                else:
                    plot_figure(ax)      
            else:
                plot_figure(ax)                
            plt.suptitle(' {} file {} signal'.format(file_name, tt), y=1.00)
            plt.savefig(mypath + "/{} file {} signal.pdf".format(file_name,tt)) 
                  
            plt.tight_layout()
            
    
    eeg_signal_name, eeg_signal,signal_channels = EEG_Signal(EEG_signal_name,EEG_signals,signal_channels)
    
    movement1= movement[0:len(eeg_signal)]  
    movement2 =movement[len(eeg_signal):]
    Signal_all =[]
    interval =np.arange(len(movement1))
    for z,m,tt in zip(interval,eeg_signal, eeg_signal_name):

        Signal = filtering(m)
        Signal_all.append(Signal)
        # plot_all() # print out all segment of the same channel 
        # plot_segment_by_segment()  
    # plot_each_segment_of_signal() # print out the segment of the all channels
    # averaging_over_walks(0.5,1.5) # the first number refers to the time before movement second number is the time after movement
    signal_left, signal_right,walk_name  = averaging_over_walks(0.5,1.5)
    signal_left_all_together.append(signal_left)
    signal_right_all_together.append(signal_right)
  
ss =[signal_left_all_together,signal_right_all_together]
## getting the min and max value
walk_name =['Left_leg','Right leg']
for ss,walk_name in zip(ss,walk_name):
    max_points =[]
    min_points =[]
    for a in range(len(ss[0][0])):
        each_walk=[]
        for b in range(len(ss)):
            for c in range(len(ss[b])):
                sinyal = ss[b][c][a]
                each_walk.append(sinyal)
        max_value = np.amax(each_walk)
        max_points.append(max_value)
        min_value = np.amin(each_walk)
        min_points.append(min_value)
                                       
    title_name =np.arange(0,int(len(segment)/2))
    for c, title_name,max_points,min_points  in zip(range(len(ss[0][0])),title_name,max_points,min_points):
     
        fig = plt.figure(figsize =([20, 20]))
         
        gs = gridspec.GridSpec(len(ss), 10)
        k =0 
        for i,EEG_signal_name in zip(range(len(ss)),EEG_signal_name1):
            
            if len(ss[i]) ==2 :
                p =3
                for j in range(len(ss[i])):
                    # if j ==0:
                    
                    globals()['ax'+str(k+1)]  = plt.subplot(gs[i, p:p+2])
                    globals()['ax'+str(k+1)].plot(ss[i][j][c],linewidth=0.5)
                    globals()['ax'+str(k+1)].axvline(x=(0.5*len(ss[0][0][0]))/(0.5+1.5),linewidth=0.4, c ='r', linestyle="--")
                    # plt.setp(globals()['ax'+str(k+2)].get_yticklabels(), visible=False)
                    globals()['ax'+str(k+1)].set_ylim(min_points,max_points)
                    plt.title('{}'.format(EEG_signal_name[j]))
                    p =p+2

                        
            if len(ss[i]) ==3 :
                for j in range(len(ss[i])):
  
                    globals()['ax'+str(k+1)]  = plt.subplot(gs[i, (j+1)*2:j*2+4])
                    globals()['ax'+str(k+1)].plot(ss[i][j][c],linewidth=0.5)
                    globals()['ax'+str(k+1)].axvline(x=(0.5*len(ss[0][0][0]))/(0.5+1.5),linewidth=0.4, c ='r', linestyle="--")
                    # plt.setp(globals()['ax'+str(k+2)].get_yticklabels(), visible=False)
                    globals()['ax'+str(k+1)].set_ylim(min_points,max_points)
                    plt.title('{}'.format(EEG_signal_name[j]))
               
            if len(ss[i]) ==4 :
                t =1
                for j in range(len(ss[i])):
  
                    globals()['ax'+str(k+1)]  = plt.subplot(gs[i, t:t+2])
                    globals()['ax'+str(k+1)].plot(ss[i][j][c],linewidth=0.5)
                    globals()['ax'+str(k+1)].axvline(x=(0.5*len(ss[0][0][0]))/(0.5+1.5),linewidth=0.4, c ='r', linestyle="--")
                    # plt.setp(globals()['ax'+str(k+2)].get_yticklabels(), visible=False)
                    globals()['ax'+str(k+1)].set_ylim(min_points,max_points)
                    plt.title('{}'.format(EEG_signal_name[j])) 
                    t =t+2
            if len(ss[i]) ==5 :
                for j in range(len(ss[i])):
                    globals()['ax'+str(k+1)]  = plt.subplot(gs[i, j*2:j*2+2])
                    globals()['ax'+str(k+1)].plot(ss[i][j][c],linewidth=0.5)
                    globals()['ax'+str(k+1)].axvline(x=(0.5*len(ss[0][0][0]))/(0.5+1.5),linewidth=0.4, c ='r', linestyle="--")
                    globals()['ax'+str(k+1)].set_ylim(min_points,max_points)
                    # plt.setp(globals()['ax'+str(k+1)].get_yticklabels(), visible=False)
                    plt.title('{}'.format(EEG_signal_name[j]))
                    # ax[k].set_ylabel('Amplitude', labelpad = 0, fontsize = 12)
                    k =k+1
                # ax1.yaxis.set_visible(True)
                fig.suptitle(' {} walk {}'.format(ordinal(title_name+1),walk_name), size=20)
                
                gs.tight_layout(fig)
            plt.tight_layout()
            plt.savefig(mypath + "/{} walk {}.pdf".format(ordinal(title_name+1),walk_name))
   

# left_leg_walk5= np.hstack((left_leg_movement[0][8],left_leg_movement[0][9]))
# np.savetxt('left_leg_walk5.txt',left_leg_walk5)

# right_leg_walk5= np.hstack((right_leg_movement[0][8],right_leg_movement[0][9]))
# np.savetxt('right_leg_walk5.txt',right_leg_walk5)

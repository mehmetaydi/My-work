'''
Created on 29 Jun 2020

@author: oskar, mehmet
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def initiate(kipale):
    df = pd.read_csv(kipale, sep=" ", header=None)
    df.columns = ["x_stock", "y_stock", "label", "additional"]
    x_values = df["x_stock"]
    y_values = df["y_stock"]
    label_values = df["label"]
    
    T1s = df.loc[df['label'] == 1]
    noT1s = df.loc[df['label'] == 0]
    
    xedges = [0,1,3,5]
    yedges = [0,2,3,4,6]
    # create edges of bins
    T1x_values = T1s["x_stock"]
    T1y_values = T1s["y_stock"]
    T1label_values = T1s["label"]
    
    noT1x_values = noT1s["x_stock"]
    noT1y_values = noT1s["y_stock"]
    noT1label_values = noT1s["label"]
    
    print(noT1x_values)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    counts, xedges, yedges, im = axes.flat[0].hist2d(T1x_values, T1y_values, range=[[-100, 100], [-100, 100]],bins=50, density=True, cmap='RdBu_r')
    counts, xedges, yedges, im = axes.flat[1].hist2d(noT1x_values, noT1y_values, range=[[-100, 100], [-100, 100]],bins=50, density=True, cmap='RdBu_r')
    #areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
    
    fig.suptitle('PCA-component adjusted probability density map')
    axes.flat[0].set_title('T1 probability density')
    axes.flat[0].set_ylabel('y')
    axes.flat[0].set_xlabel('x')
    
    
    axes.flat[1].set_title('no-T1 probability density')
    #axes.flat[1].set_ylabel('-0.89*angle1 - 0.28*angle2 + 0.35*angle3')
    axes.flat[1].set_xlabel('x')
    
    
    plt.colorbar(im, ax=axes.flat[1])
    #im = axes.flat[0].imshow(H, interpolation='nearest', origin='low')
    
    #plt.xticks((0, 20, 40, 60, 80, 100, 120, 140), ('-200','-150','-100','-50','0','50','100','150','200'))

    
    #interpolation='nearest', origin='low'
    #H, xedges, yedges = np.histogram2d(noT1x_values, noT1y_values, range=[[-300, 300], [-300, 300]],bins=150, density=True)
    #areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
    #im = axes.flat[1].imshow(H, interpolation='nearest', origin='low')
    #extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    #plt.xticks(np.arange(-250, 250, 50)) 
    #plt.yticks(np.arange(-250, 250, 50)) 
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
    
    
    
#initiate('adjusted_scaled_scaled.txt')
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:47:13 2022

@author: noise

neural network plot utilitiy functions
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotFullFields(Enn,Etest, epoch,i, dir):
    
    if not os.path.isdir(dir+str(i)+'/'):
        os.makedirs(dir+str(i)+'/')
    Enn = Enn.detach().cpu().numpy()  
    vmax = np.max(np.abs(Etest)**2)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.suptitle('epoch = '+str(epoch), x=0.5, y=0.75, fontsize=16)
    
    axes[0].set_title('$||E_{nn}||^2$')
    im1 = axes[0].imshow(np.abs(Enn)**2, cmap='hot', vmin=0, vmax=vmax)
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    
    axes[1].set_title('||$E_{test}||^2$')
    im2 = axes[1].imshow(np.abs(Etest)**2, cmap='hot', vmin=0, vmax=vmax)
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    axes[2].set_title('$||E_{nn}-E_{test}||^2$')
    im3 = axes[2].imshow(np.abs(Enn-Etest)**2, cmap='hot', vmin=0, vmax=vmax)
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    
    #plt.subplots_adjust(top=0.75)
    plt.tight_layout()
    
    plt.savefig(dir+str(i)+'/'+str(epoch)+".png")
    plt.close()
    
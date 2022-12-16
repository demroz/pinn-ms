#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:32:59 2022

@author: noise
"""

import numpy as np
import matplotlib.pyplot as plt 
figDir = '/home/noise/Documents/pinn-paper/figs/texFigs/'
savedir = 'nnTrainingData-aug8/'
train_loss_ar = np.loadtxt(savedir+'train_loss')
test_loss_ar = np.loadtxt(savedir+'test_loss' )
test_norm_error_ar = np.loadtxt(savedir+'fro_loss')

i = np.arange(0,len(train_loss_ar))
j = np.arange(0,len(train_loss_ar),1000)

fig,ax = plt.subplots()
ax.semilogy(i,train_loss_ar)
ax.semilogy(j,test_loss_ar)
plt.legend(['training loss', 'test loss'])
#plt.xlabel('iteration')
#plt.ylabel('loss')
plt.rcParams.update({'font.size': 15})
plt.savefig(figDir+'/NNLoss.png', dpi=300, bbox_inches = "tight")


fig,ax = plt.subplots()
ax.semilogy(j,test_norm_error_ar)
ax.set_ylim([0,10**1])
#plt.xlabel('iteration')
#plt.ylabel(r'$\frac{|E_{nn}-E_{fdfd}|_1}{|E_{fdfd}|_1}$')
plt.rcParams.update({'font.size': 15})
plt.savefig(figDir+'/FrobLoss.png', dpi=300,bbox_inches = "tight")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:33:00 2022

@author: noise
"""

import numpy as np
import matplotlib.pyplot as plt 

def generateLens(f, lensD):
    
    #lensD = 50
    data = np.loadtxt('/home/noise/code/pinn/inv-design-2d/pinn-metagrating-optimization/generateLPAData/lpaData/r_v_phi633')
    r = data[0]
    phi = data[1]
    
    p = 0.443
    wave = 0.633
    x1 = np.arange(0,lensD/2,p)
    x2 = -np.flip(x1[1:-1])
    x = np.concatenate([x2,x1])
    phaseProfile = 2*np.pi/wave*(np.sqrt(x**2+f**2)-f)%(2*np.pi) # np.mod(2*np.pi/wave*(f-np.sqrt(x**2+f**2)),2*np.pi)
    
    plt.figure()
    plt.plot(x, phaseProfile)
    plt.show()
    
    lensRadArray = []
    lensPhaseArray = []
    for i in range(0,len(x)):
        nearestRadiusIndex = np.argmin(np.abs(phi-phaseProfile[i]))
        nearestRadiusPhase = phi[nearestRadiusIndex]
        nearestRadius = r[nearestRadiusIndex]
        
        lensRadArray.append(nearestRadius)
        lensPhaseArray.append(nearestRadiusPhase)
        
    return x, lensRadArray, phaseProfile

x,r,p = generateLens(50, 50)
np.savetxt('xlocs_fwd.dat',x)
np.savetxt('r_fwd.dat',r)
plt.figure()
plt.plot(x,p)
plt.show()
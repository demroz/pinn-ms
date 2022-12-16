#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:33:00 2022

@author: noise
"""

import numpy as np
import matplotlib.pyplot as plt 
from torchPhysicsUtils import *

def generateLensFromPhase(x, phase):
    data = np.loadtxt('/home/noise/code/pinn/inv-design-2d/fwdDesign2D/lpaData/r_v_phi550')
    r = data[0]
    phi = data[1]
    
    lensRadArray = np.zeros(len(phase))
    lensPhaseArray = np.zeros(len(phase))
    
    phaseProfile = phase
    
    for i in np.arange(0,len(x),1):
        nearestRadiusIndex = np.argmin(np.abs(phi-phaseProfile[i]))
        nearestRadiusPhase = phi[nearestRadiusIndex]
        nearestRadius = r[nearestRadiusIndex]
        
        lensRadArray[i] = nearestRadius
        lensPhaseArray[i] = nearestRadiusPhase
        
    
    plt.figure()
    plt.plot(phaseProfile)
    plt.plot(lensPhaseArray)
    plt.show()
    return x, lensRadArray

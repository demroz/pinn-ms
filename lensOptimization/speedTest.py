#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:44:49 2022

@author: noise
"""


from optimizeLensMemoryOptimizedGPU import *
from generateLens import *
from computeEfficiencies import *

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from optimizeLensMemoryOptimizedGPU import *

dlist = [1000]
tlist = []
for D in dlist:
    f = 250
    #D = 100
    x,r, phase = generateLens(f, D)
    r = np.clip(r,0.075/2, 0.443/2-0.075/2)
    NA = np.sin(np.arctan(D/(2*f)))
    # #%%
    rfwd = np.flip(r[0:int(len(r)/2)])
    xfwd = x[int(len(r)/2):len(r)]
    rfwd_sim = np.concatenate([np.flip(rfwd),rfwd])
    xfwd_sim = np.concatenate([-np.flip(xfwd),xfwd])
    zmax = 1000
    dz = dx*50
    rOptGDS, result, time = optGaussianProfileADAM(rfwd, f, 9)
    tlist.append(time)
   #%% 
#np.savetxt('/home/noise/Dropbox/paperfigs/speed.dat',[dlist,tlist])
from angler import Simulation

import time

Ezl = []

bR = rVectorToBatchRadii(torch.tensor(r), 11, 9)
patches = batchToEpsPatch(bR).numpy()
st = time.time() 
batchToPatches(bR)
et = time.time()
st = time.time() 
batchToPatches(bR)
et = time.time()
print('nn simtime',st-et)
#%%
for i in range(patches.shape[0]):
    st = time.time()    
    sim = Simulation(2*np.pi/0.633,patches[i],0.443/16,[10,10],'Ez')
    sim.src[15,:] = 10;
    et = time.time()
    print(et-st)
    Ez,_,_ = sim.solve_fields()
    Ezl.append(Ez)
    
result = stitchPatches(torch.tensor(Ezl), 9)
#%%
from ceviche import fdfd_ez
Ezl = []

bR = rVectorToBatchRadii(torch.tensor(r), 11, 9)
patches = batchToEpsPatch(bR).numpy()
st = time.time() 
batchToPatches(bR)
et = time.time()
print('nn simtime',st-et)
for i in range(patches.shape[0]):
    st = time.time()    
    
    sim = fdfd_ez(2*np.pi/0.633,0.443/16,patches[i],[10,10])
    source = np.zeros(patches[i].shape)
    source[15,:] = 10;
    et = time.time()
    print(et-st)
    Ez,_,_ = sim.solve(source)
    Ezl.append(Ez)
    
result = stitchPatches(torch.tensor(Ezl), 9)

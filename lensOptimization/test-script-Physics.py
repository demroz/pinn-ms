#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 13:30:28 2022

@author: noise
"""

import torch
import matplotlib.pyplot as plt
from torchPhysicsUtils import *

# generate lens phase profile
f = 50
D = 50
p = 0.443
Np = 16
dx = p/Np
wave = 0.633
omega = 2*np.pi/wave
x = np.arange(-D/2,D/2,dx)
phaseProfile = torch.tensor(np.mod(2*np.pi/wave*(f-np.sqrt(x**2+f**2)),2*np.pi))
nearField = torch.exp(1j*phaseProfile).cuda()

#%% test angular spectrum propagation

zlist = np.arange(5,100,1)

I = np.zeros([len(phaseProfile),len(zlist)])
for i in range(len(zlist)):
    p = Propagator1DPadded(len(nearField), omega, zlist[i], dx, pad_factor=3., device = torch.device('cuda'))
    ff = p.prop(nearField)
    I[:,i] = torch.abs(ff).cpu().detach().numpy()**2
    
plt.figure()
plt.imshow(I)
plt.colorbar()
plt.gca().set_aspect(0.05)
plt.show()
#%% test r-s propt
xnear = torch.tensor(x).cuda()
rsp = RayleighSommerfieldPropagation(omega, dx, device = torch.device('cuda'))
ff = rsp.propagate(nearField, xnear, 0, 50)

ffarr = []

for i in range(len(xnear)):
    ffarr.append(torch.abs(rsp.propagate(nearField, xnear, x[i], 50)).cpu().detach().numpy()**2)
    
plt.figure()
plt.plot(xnear.cpu(),ffarr)
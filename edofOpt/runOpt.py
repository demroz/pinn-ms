#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 11:28:40 2022

@author: noise
"""

from phaseOptimization import *
from forwardDesignLens import *
#%%
f0 = 500
f1 = 1000
D = 500
x = np.arange(-D/2, D/2, period)
#%%
initPhase = np.random.uniform(size=len(x))
phase = optimizePhase(initPhase, f0, f1)

#%%
_, r  = generateLensFromPhase(x, phase)
np.savetxt('optphase.dat',[x,r])
#x,r = np.loadtxt('optphase.dat')
#%%
rinit = np.random.uniform(size=r[int(len(r)/2):len(r)].shape)*0.443/4
from optimizeLensMemoryOptimizedGPU import *
ropt_half, result = optEDOFAdam(rinit, f0, f1, 3)
ropt = np.concatenate([np.flip(ropt_half),ropt_half])

#%% 
x = np.concatenate([-np.flip(x[int(len(r)/2):len(r)]),x[int(len(r)/2):len(r)]])
from computeFarfields import *
Ifwd, Efwd, zlist = computeLensFarfields(r, x, f1, D, 0.633, f1+30)
Iinv, Einv, zlist = computeLensFarfields(ropt, x, f1, D, 0.633, f1+30)

#%%
plt.figure()
plt.subplot(2,1,1)
plt.imshow(Ifwd)
plt.gca().set_aspect(0.05)
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(Iinv)
plt.gca().set_aspect(0.05)
plt.colorbar()
#%%
xgrid = np.arange(-Ifwd.shape[0]/2,Ifwd.shape[0]/2, 1)*0.443/16
from computeFarfields import *
eff_f = computeEfficiencyAtZ(Ifwd, Efwd, xgrid, zlist, 75)
eff_inv = computeEfficiencyAtZ(Iinv, Efwd, xgrid, zlist, 75)

#%%
eff_f = []
eff_i = []

for z in np.linspace(f0,f1,50):
    eff_f.append(computeEfficiencyAtZ(Ifwd, Efwd, xgrid, zlist, z))
    eff_i.append(computeEfficiencyAtZ(Iinv, Einv, xgrid, zlist, z))

plt.figure()
plt.plot(eff_f)
plt.plot(eff_i)
plt.legend(['fwd','inv'])
plt.show()
#%% maxi calc

imax_f = []
imax_i = []
for i in range(0,Ifwd.shape[1]):
    imax_f.append(np.max(Ifwd[:,i]))
    imax_i.append(np.max(Iinv[:,i]))
    
plt.figure()
plt.plot(imax_f)
plt.plot(imax_i)
plt.show()
#%%
z = 1000
iz = np.argmin(np.abs(zlist-z))
plt.plot(zlist,Iinv[18243,:])
plt.plot(zlist,Ifwd[18243,:])   
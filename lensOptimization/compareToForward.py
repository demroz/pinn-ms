#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 21:19:20 2022

@author: noise
"""

from optimizeLensMemoryOptimizedGPU import *
from generateLens import *
from computeEfficiencies import *

import numpy as np
import matplotlib.pyplot as plt

# best 1400 with gap
flen = [200,400,600,800,1000,1200,1400,1600,1800, 2000]
#%%
def plotfwdandinv(fwd,inv, f):
    Ifwd = fwd['I']
    Iinv = inv['I']
    
    imax_f, jmax_f = np.unravel_index(np.argmax(Ifwd),Ifwd.shape)
    imax_i, jmax_i = np.unravel_index(np.argmax(Iinv),Iinv.shape)
    
    zmax = 2200 # microns
    dx = 0.443/16
    dz = dx*50 #0.01 #0.443/16
     
    zlist = np.arange(0,zmax,dz)
    print(zlist.shape)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(Ifwd)
    plt.gca().set_aspect(0.5)
    plt.colorbar()
    plt.title('fwd')
    plt.subplot(2,1,2)
    plt.imshow(Iinv)
    plt.gca().set_aspect(0.5)
    plt.colorbar()
    plt.title('inv')
    #plt.savefig('data/I-'+str(f)+'um.png')
    plt.show()
    plt.close()
    fspot_f = Ifwd[:,jmax_f]
    fspot_i = Iinv[:,jmax_i]
    print(jmax_f, jmax_i)
    xgrid = fwd['xgrid']
    
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    plt.plot(xgrid[np.abs(xgrid) < 5], fspot_f[np.abs(xgrid) < 5])
    plt.ylim([0,20])
    plt.text(0.55, 0.95, "Max: {:.3f}".format(np.max(fspot_f)), fontweight="bold", transform=ax1.transAxes)
    plt.text(0.55, 0.9, "$\eta$: {:.3f}".format(fwd['efficiency']), fontweight="bold", transform=ax1.transAxes)
    plt.text(0.55, 0.85, "F: {:.3f}".format(zlist[jmax_f]), fontweight="bold", transform=ax1.transAxes)
    plt.title('fwd')
    
    ax2 = plt.subplot(1,2,2)
    plt.plot(xgrid[np.abs(xgrid) < 5], fspot_i[np.abs(xgrid) < 5])
    plt.text(0.55, 0.95, "Max: {:.3f}".format(np.max(fspot_i)), fontweight="bold", transform=ax2.transAxes)
    plt.text(0.55, 0.9, "$\eta$: {:.3f}".format(inv['efficiency']), fontweight="bold", transform=ax2.transAxes)
    plt.text(0.55, 0.85, "F: {:.3f}".format(zlist[jmax_i]), fontweight="bold", transform=ax2.transAxes)
    
    plt.ylim([0,20])
    plt.title('inverse')
    #plt.savefig('data/focal-spots-'+str(f)+'um.png')
    plt.show()
    plt.close()

#%%   i
D = 1000
eff_f = []
eff_i = []
for f in flen:
    import numpy as np
    x,r, phase = generateLens(f, D)
    r = np.clip(r,dx, 0.443/2)
    NA = np.sin(np.arctan(D/(2*f)))
    # #%%
    rfwd = np.flip(r[0:int(len(r)/2)])
    xfwd = x[int(len(r)/2):len(r)]
    rfwd_sim = np.concatenate([np.flip(rfwd),rfwd])
    xfwd_sim = np.concatenate([-np.flip(xfwd),xfwd])
    #%%
    fwdLensDict = computeLensEfficiency(rfwd_sim, xfwd_sim, f, D)
    #%%
    from optimizeLensMemoryOptimizedGPU import *
    rOptGDS, result = optGaussianProfileADAM(rfwd, f, 9)
    #%%
    rInv = np.concatenate([np.flip(rOptGDS),rOptGDS])
    xInv = np.concatenate([-np.flip(xfwd),xfwd])
    #%%
    invLensDict = computeLensEfficiency(rInv, xInv, f, D)
    #%%
    plotfwdandinv(fwdLensDict,invLensDict,f) 
 #%%   
    eff_f.append(fwdLensDict['efficiency'])
    eff_i.append(invLensDict['efficiency'])

#%%
plt.figure()
plt.plot(flen, eff_f)
plt.plot(flen, eff_i)
plt.show()
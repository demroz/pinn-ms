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
#%%
def plotfwdandinv(fwd,inv, f):
    Ifwd = fwd['I']
    Iinv = inv['I']
    
    imax_f, jmax_f = np.unravel_index(np.argmax(Ifwd),Ifwd.shape)
    imax_i, jmax_i = np.unravel_index(np.argmax(Iinv),Iinv.shape)
    
    zmax = 1000 # microns
    dx = 0.443/16
    dz = dx*50 #0.01 #0.443/16
     
    zlist = np.arange(0,zmax,dz)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(np.log(Ifwd),extent=[0,1000,-500,500])
    #plt.gca().set_aspect(0.005)
    plt.colorbar()
    plt.title('fwd')
    plt.subplot(2,1,2)
    plt.imshow(np.log(Iinv),extent=[0,1000,-500,500])
    #plt.gca().set_aspect(0.005)
    plt.colorbar()
    plt.title('inv')
    #plt.savefig('data/I-'+str(f)+'um.png')
    plt.tight_layout()
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
import numpy as np
f = 250
D = 1000
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
#%%
fwdLensDict = computeLensEfficiency(rfwd_sim, xfwd_sim, f, D, 500, dz/50)
#%%
from optimizeLensMemoryOptimizedGPU import *
rOptGDS, result = optGaussianProfileADAM(rfwd, f, 9)
#%%
rInv = np.concatenate([np.flip(rOptGDS),rOptGDS])
xInv = np.concatenate([-np.flip(xfwd),xfwd])
#%%
invLensDict = computeLensEfficiency(rInv, xInv, f, D, 500, dz/50)
#%%
plotfwdandinv(fwdLensDict,invLensDict,f)    
#%%
#np.savetxt('designedLensesSept13/fwd-50nmGap-f'+str(f)+'.dat', [xfwd_sim, rfwd_sim])
#np.savetxt('designedLensesSept13/inv-50nmGap-f'+str(f)+'.dat', [xInv, rInv])

#%%
imin= int(fwdLensDict['I'].shape[0]/2-180)
imax= int(fwdLensDict['I'].shape[0]/2+180)
jmin= int(fwdLensDict['I'].shape[1]/2-300)
jmax= int(fwdLensDict['I'].shape[1]/2+300)

im = fwdLensDict['I'][imin:imax,jmin:jmax]
plt.figure()
img = plt.imshow(im)
plt.gca().set_aspect(0.8)
#plt.colorbar()
plt.colorbar(img, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.xticks([])
plt.yticks([])

#%%
im = invLensDict['I'][imin:imax,jmin:jmax]
plt.figure()
img = plt.imshow(im)
plt.gca().set_aspect(0.8)
#plt.colorbar()
plt.colorbar(img, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.xticks([])
plt.yticks([])

#%%
imax_f, jmax_f = np.unravel_index(np.argmax(invLensDict['I']),invLensDict['I'].shape)
xgrid = fwdLensDict['xgrid']
plt.figure()
plt.plot(xgrid[imin:imax],fwdLensDict['I'][imin:imax,jmax_f],'b')
plt.plot(xgrid[imin:imax],invLensDict['I'][imin:imax,jmax_f],'r')
plt.legend(['forward', 'inverse'])
plt.savefig('/home/noise/Dropbox/paperfigs/fspots.png',dpi=500)

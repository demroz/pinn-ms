#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:09:24 2022

@author: noise
"""

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
def plotfwdandinv(fwd,inv, f, D,zmax,dz):
    savedir = 'optimizationData/'
    Ifwd = fwd['I']
    Iinv = inv['I']
    
    imax_f, jmax_f = np.unravel_index(np.argmax(Ifwd),Ifwd.shape)
    imax_i, jmax_i = np.unravel_index(np.argmax(Iinv),Iinv.shape)
    
    #zmax = 200 # microns
    #dx = 0.443/16
    #dz = dx #0.01 #0.443/16
     
    zlist = np.arange(0,zmax,dz)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(Ifwd, extent = [0,zmax,-D/2,D/2])
    #plt.gca().set_aspect(0.01)
    plt.colorbar()
    plt.title('fwd')
    plt.subplot(2,1,2)
    plt.imshow(Iinv,extent = [0,zmax,-D/2,D/2])
    #plt.gca().set_aspect(0.01)
    plt.colorbar()
    plt.title('inv')
    plt.tight_layout()
    plt.savefig(savedir+'I-'+str(f)+'um.png')
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
    plt.tight_layout()
    plt.savefig(savedir+'focal-spots-'+str(f)+'um.png',dpi=500)
    
    plt.show()
    plt.close()

#%%
numrads = 9
from optimizeLensMemoryOptimizedGPU import *
flist = [250,500,750,1000,1250,1500]
D = 1000
zmax = 1700
dz = dx*10

fwd_eff = []
fwd_imax = []

inv_eff = []
inv_imax = []
import time
for f in flist:
    x,r, phase = generateLens(f, D)
    r = np.clip(r,0.075/2, 0.443/2-0.075/2)
    NA = np.sin(np.arctan(D/(2*f)))
    # #%%
    rfwd = np.flip(r[0:int(len(r)/2)])
    xfwd = x[int(len(r)/2):len(r)]
    rfwd_sim = np.concatenate([np.flip(rfwd),rfwd])
    xfwd_sim = np.concatenate([-np.flip(xfwd),xfwd])
    fwdLensDict = computeLensEfficiency(rfwd_sim, xfwd_sim, f, D, zmax, dz)
    st = time.time()
    rOptGDS, result = optGaussianProfileADAM(rfwd, f, numrads)
    et = time.time()
    print('1 iteration takes', et-st)
    rInv = np.concatenate([np.flip(rOptGDS),rOptGDS])
    xInv = np.concatenate([-np.flip(xfwd),xfwd])
    invLensDict = computeLensEfficiency(rInv, xInv, f, D, zmax, dz)
    #%%
    plotfwdandinv(fwdLensDict,invLensDict,f,D,zmax, dz)   
    #%%
    fwd_eff.append(fwdLensDict['efficiency'])
    fwd_imax.append(np.max(fwdLensDict['I']))
    inv_eff.append(invLensDict['efficiency'])
    inv_imax.append(np.max(invLensDict['I']))
    np.savetxt('designedLensesOct2/fwd-50nmGap-f'+str(f)+'.dat', [xfwd_sim, rfwd_sim])
    np.savetxt('designedLensesOct2/inv-50nmGap-f'+str(f)+'.dat', [xInv, rInv])
    
#%%

pltdir = '/home/noise/Documents/pinn-paper/figs/texFigs/'
NA = np.sin(np.arctan(D/(2*np.array(flist))))
np.savetxt('optimizationData/fwd_eff'+str(numrads)+'.dat',fwd_eff)
np.savetxt('optimizationData/fwd_imax'+str(numrads)+'.dat',fwd_imax)
np.savetxt('optimizationData/inv_eff'+str(numrads)+'.dat',inv_eff)
np.savetxt('optimizationData/inv_imax'+str(numrads)+'.dat',inv_imax)
#%%
plt.rcParams.update({'font.size': 14})
plt.figure()
plt.plot(NA,fwd_eff,'b-.')
plt.plot(NA,inv_eff,'r-.')
#plt.xlabel('NA')
#plt.legend(['forward','inverse'])
plt.tight_layout()
plt.savefig('/home/noise/Dropbox/paperfigs/efficiency.png',dpi=500)
plt.show()
#%%
plt.figure()
plt.plot(NA,fwd_imax/np.max(fwd_imax),'b-.')
plt.plot(NA,inv_imax/np.max(fwd_imax),'r-.')
plt.legend(['forward','inverse'])
plt.tight_layout()
plt.savefig('/home/noise/Dropbox/paperfigs/imax.png',dpi=500)
#plt.xlabel('NA')
#plt.ylabel('max(I)')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:11:29 2022

@author: noise
"""

import pickle
import numpy as np
with open('fwd-lens-2050um.pkl', 'rb') as f:
    fwd = pickle.load(f)
    
with open('inv-lens-2050um.pkl', 'rb') as f:
    inv = pickle.load(f)
    
zmax = 2500
dx = 0.443/16
dz = dx*10

x = np.arange(-500,500,dx)
z = np.arange(0,zmax,dz)
#%%
import scipy.io as sio
sio.savemat('/home/noise/Dropbox/paperfigs/fwdedof.mat',{'I':fwd['I']})
sio.savemat('/home/noise/Dropbox/paperfigs/invedof.mat',{'I':inv['I']})

#%%
import matplotlib.pyplot as plt

imin = int(fwd['I'].shape[0]/2)-2000
imax = int(fwd['I'].shape[0]/2)+2000
jmin = int(1900/dz)
jmax = int(2200/dz)#fwd['I'].shape[1]
i0 = int(fwd['I'].shape[0]/2)
#%%
plt.figure()
plt.pcolormesh(z[jmin-1:jmax],x[imin-1:imax]-5, fwd['I'][imin:imax,jmin:jmax])
plt.xticks([1900,2050,2200])
plt.yticks([-25,0,25])
plt.savefig('/home/noise/Dropbox/paperfigs/fwd2050.png',dpi=500)
plt.show()

plt.figure()
plt.pcolormesh(z[jmin-1:jmax],x[imin-1:imax]-5, inv['I'][imin:imax,jmin:jmax]/np.max(inv['I'][imin:imax,jmin:jmax]))
#plt.colorbar()
plt.xticks([1900,2050,2200])
plt.yticks([-25,0,25])
plt.savefig('/home/noise/Dropbox/paperfigs/inv2050.png',dpi=500)
plt.show()
#%%
jmin = int(1900/dz)
jmax = int(2200/dz)

j0 = int(2025/dz)
j1 = int(2055/dz)
j2 = int(2075/dz)

plt.figure()
plt.plot(z[jmin:jmax],fwd['I'][i0,jmin:jmax])
plt.plot(z[jmin:jmax],inv['I'][i0,jmin:jmax])
plt.show()
#%%
imin = int(fwd['I'].shape[0]/2)-1000
imax = int(fwd['I'].shape[0]/2)+1000
from scipy import special
xx = x[imin:imax]
k = 2*np.pi/0.633
a = 500
theta = np.arctan((xx-5)/2025)
n = 1
airy = (2*sp.special.j1(k*a*np.sin(theta))/(k*a*np.sin(theta)))**2
plt.rc('font',size=12)
plt.figure()
plt.plot(xx-4.99231,inv['I'][imin:imax:n,j0]/np.max(inv['I'][imin:imax:n,j0]),'r-.')
plt.plot(xx-4.99231,airy,'b')
plt.xticks([-25,25])
plt.yticks([0,0.5,1])
plt.savefig('/home/noise/Dropbox/paperfigs/airy2025.png',dpi=500)

plt.show()

plt.figure()
plt.plot(xx-4.99231,inv['I'][imin:imax:n,j1]/np.max(inv['I'][imin:imax:n,j1]),'r-.')
plt.plot(xx-4.99231,airy,'b')
plt.xticks([-25,25])
plt.yticks([0,0.5,1])
plt.legend(['EDOF','Perfect Lens'])
plt.savefig('/home/noise/Dropbox/paperfigs/airy2050.png',dpi=500)
plt.show()

plt.figure()
plt.plot(xx-4.99231,inv['I'][imin:imax:n,j2]/np.max(inv['I'][imin:imax:n,j2]),'r-.')
plt.plot(xx-4.99231,airy,'b')
plt.xticks([-25,25])
plt.yticks([0,0.5,1])
plt.savefig('/home/noise/Dropbox/paperfigs/airy2075.png',dpi=500)
plt.show()

#%%
n = 15
inv_x0 = inv['I'][i0,jmin:jmax:n]/np.max(inv['I'][i0,jmin:jmax:n])
fwd_x0 = fwd['I'][i0,jmin:jmax:n]/np.max(fwd['I'][i0,jmin:jmax:n])
zz = z[jmin:jmax:n]
idx = np.argwhere(np.diff(np.sign(inv_x0 - 0.5))).flatten()
idx = np.argwhere(np.diff(np.sign(fwd_x0 - 0.5))).flatten()


plt.figure()
plt.plot(zz,inv_x0,'r-.')
plt.plot(zz,fwd_x0,'b--')
plt.xticks([1900,2050,2200])
plt.yticks([0,0.5,1])
plt.savefig('/home/noise/Dropbox/paperfigs/dof.png',dpi=500)

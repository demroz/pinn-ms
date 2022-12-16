#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:55:32 2022

@author: noise
"""
import numpy as np
from computeEfficiencies import *
import numpy as np
import matplotlib.pyplot as plt
from torchPhysicsUtils import *
datafileinv = '/home/noise/code/pinn/inv-design-2d/pinn-metagrating-optimization/lensOptimization/designedLensesOct2/inv-50nmGap-f500.dat'
datafilefwd = '/home/noise/code/pinn/inv-design-2d/pinn-metagrating-optimization/lensOptimization/designedLensesOct2/fwd-50nmGap-f500.dat'

x,r = np.loadtxt(datafilefwd)
x1,r1 = np.loadtxt(datafileinv)
dz = 0.443/16*10
#fwdLensDict = computeLensEfficiency(r, x, 250, 1000, 500, dz)
#%%
Ef = simulateLens(r, x, 250, 1000)
Ei = simulateLens(r1, x1, 250, 1000)
#%%
dx = 0.443/16
zmin = 250
zmax = 750
diam = 1000
omega = 2*np.pi/0.633
xgrid = np.arange(-diam/2-5,diam/2+5,dx)
zlist = np.arange(zmin,zmax,dz)

Etf = np.conj(Ef[65,:])
Eti = np.conj(Ei[65,:])

Efa = np.zeros([len(Etf),len(zlist)],dtype=np.complex)
Eia = np.zeros([len(Etf),len(zlist)],dtype=np.complex)
for i in range(0,len(zlist)):
    pr = Propagator1DPadded(len(Etf), omega, zlist[i], dx, pad_factor=3., device = torch.device('cpu'))
    fieldAtFocalPlanef = pr.prop(torch.tensor(Etf))
    fieldAtFocalPlanei = pr.prop(torch.tensor(Eti))
    Efa[:,i] = fieldAtFocalPlanef.numpy()
    Eia[:,i] = fieldAtFocalPlanei.numpy()
plt.figure()
plt.imshow(np.abs(Efa)**2)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(Eia)**2)
plt.colorbar()
#%%%
# imin = int(fwdLensDict['I'].shape[0]/2-100)
# imax = int(fwdLensDict['I'].shape[0]/2+100)
# jmin = int(fwdLensDict['I'].shape[1]/2-300)
# jmax = int(fwdLensDict['I'].shape[1]/2+300)
# plt.figure()
# plt.pcolormesh(fwdLensDict['I'][imin:imax,jmin:jmax])#,vmin=0,vmax=4)
# plt.gca().set_aspect(0.35)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.savefig('/home/noise/Dropbox/paperfigs/250uminv.png',dpi=500)
#%%
np.savetxt('/home/noise/Dropbox/paperfigs/paperdata/500fwd.dat',np.abs(Efa)**2)
np.savetxt('/home/noise/Dropbox/paperfigs/paperdata/500inv.dat',np.abs(Eia)**2)
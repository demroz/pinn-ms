#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:09:21 2022

@author: noise
"""
import numpy as np
from computeEfficiencies import *
import matplotlib.pyplot as plt

fwdLens = np.loadtxt('designedLenses/fwdDesignedLens50nmgap1000umF.dat')
invLens = np.loadtxt('designedLenses/optimizedLens50nmgap1000umF.dat')

xf = fwdLens[0]
rf = fwdLens[1]

xi = invLens[0]
ri = invLens[1]

import meep as mp

h = 0.6
p = 0.443
Np = 16
res = 1/(p/Np)
wave = 0.633
cell = mp.Vector3(1010,5,0)

geometry = []
geometry.append(mp.Block(mp.Vector3(1000,2.5,mp.inf),
                         center=mp.Vector3(0,-(2.5/2+h/2),0),
                         material=mp.Medium(epsilon=1.4**2)))

for i in range(len(xi)):
    geometry.append(mp.Block(mp.Vector3(2*ri[i],h,mp.inf),
                     center=mp.Vector3(xi[i],0,0),
                     material=mp.Medium(epsilon=4)))

sources = [mp.Source(mp.ContinuousSource(frequency=1/wave),
                     component=mp.Ez,
                     center=mp.Vector3(0,-1),
                     size=mp.Vector3(1010,0,0))]
pml_layers = [mp.PML(1.0)]


#%%
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=res,
                    force_complex_fields=True)
#%%
#sim.init_sim()
#%%
sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0,0),
                                  size  =mp.Vector3(10,5)))

sim.run(until=100)


Ez = sim.get_array(mp.Ez)
#%%
np.savetxt('Ezfdtd',Ez)

#%%
Et = Ez[:,130]
diam = 1000
from computeEfficiencies import *
dict = computeEfficiencyFromField(Et, diam)
plt.figure()
plt.imshow(dict['I'],extent=[0,np.max(dict['zlist']),-500,500])
plt.clim(0,1)
plt.colorbar()

#%%
plt.figure()
plt.plot(dict['I'][:,np.argmin(np.abs(dict['zlist']-1000))])
plt.title('z = 1000')
#%%
plt.figure()
plt.plot(dict['I'][:,356])
plt.title('z = 500')

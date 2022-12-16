#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:10:11 2022

@author: noise
"""
import numpy as np
import matplotlib.pyplot as plt 
from angler import Simulation
import pandas as pd

datadir = "lpaData/"

NPML = [25,0]

omega = 2*np.pi/0.633 #2*np.pi/0.633
Np = 200
p = 0.443
x = np.linspace(-p/2,p/2,Np)
y = np.linspace(-p*2,p*2,Np*4)
xx,yy = np.meshgrid(x,y)

rads = np.array([0.2])*p/2
radlocs = np.array([0])
h = 0.6
dx = x[2]-x[1]

def generateEps(r):
    eps = np.ones(xx.shape)
    for i in range(0,len(radlocs)):
        eps[(np.abs(xx-radlocs[i]) < r[i]) &
            (np.abs(yy) < h/2)
            ] = 2**2
    return eps

radlist = np.linspace(0,p/2,200)

phi = np.zeros(radlist.shape)
T = np.zeros(radlist.shape)
for i in range(0,len(radlist)):
    d = []
    rads = np.array([radlist[i]])
    
    eps = generateEps(rads)
    
    sim = Simulation(omega, eps, dx, NPML, 'Ez')
    sim.src[100,:] = 20
    sim.solve_fields()
    
    #sim.plt_re()
    #sim.plt_abs()
    
    E = sim.fields['Ez']
    #sim.plt_abs()
    #plt.show()
    Ed = E[700,:]
    angle = np.angle(Ed)
    #plt.figure()
    #plt.imshow(np.abs(E)**2)
    #plt.show()
    phi[i] = np.mean(angle)
    #EzD
#%%
plt.figure()
plt.plot(radlist,phi)#%(np.pi*2))
plt.xlabel('radius microns')
plt.ylabel('phase shift')
plt.show()
plt.savefig('lpaData/phaseplot.png')
plt.show()

# # #%%
data = np.array([radlist,phi%(2*np.pi)])
np.savetxt('lpaData/r_v_phi633',data)
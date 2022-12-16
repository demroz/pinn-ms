#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:34:25 2022

@author: noise
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:16:37 2022

@author: noise
"""
from angler import Simulation
import numpy as np
import matplotlib.pyplot as plt
from torchPhysicsUtils import *
import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
#%%
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def ssig(x, a, b):
    return 1./(1.+torch.exp(-a*(x-b)))

def mesh(radii, radlocs, diam):
    Np = 16
    p = 0.443
    #diam = 50
    dx = p/Np
    x = torch.arange(-diam/2-5,diam/2+5,dx)
    y = torch.linspace(-p*3,p*3,Np*6)
    xx,yy = torch.meshgrid(x,y)
    
    xx = xx.T
    yy = yy.T
    
    h = 0.6
    dx = x[2]-x[1]
    
    gh = torch.exp(-yy**2/(2*(h/2)**2))
    
    gl = []
    pil = torch.zeros(gh.shape)#.cuda()
    
    a = 100
    for i in range(len(radii)):
        gl.append(torch.exp(-(xx-radlocs[i])**2/(2*radii[i]**2)))
        pil += (ssig(gl[i],a,np.log(2))*ssig(gh,a,np.log(2)))
    
    return pil.cpu().numpy()*(2**2-1)+1.
#%%
def simMS(xData,radData, F):
    
    wave = 0.633
    omega = 2*np.pi/wave
    
    diam = 2*np.max(xData)
    zmax = F+100
    p = 0.443
    dx = p/16
    h = 0.6
    Np = 16
    xgrid = np.arange(-diam/2-1,diam/2+1,dx)
    #ygrid = np.linspace(-p*3,p*3, Np*6)#np.arange(-p*3,5/2,dx)
    ygrid = np.arange(-p*3,zmax,p/Np)#np.linspace(-p*3,p*3, Np*6)#np.arange(-p*3,5/2,dx)
    
    xx,yy = np.meshgrid(xgrid,ygrid)
    NPML = [20,20]
    

    eps = np.ones(xx.shape)
    for i in range(0,len(radData)):
        rad = radData[i]
        xloc = xData[i]
        eps[(np.abs(xx-xloc) < rad) &
            (np.abs(yy) < h/2)] = 2**2
        
    sim = Simulation(omega, eps, dx, NPML, "Ez")#, use_dirichlet_bcs=True)
    sim.src[25,:] = 10
    sim.solve_fields()
    sim.plt_re()
    sim.plt_abs()
    return sim.fields['Ez']

def simulateLens(radData, xData, F, diam):
    wave = 0.633
    omega = 2*np.pi/wave
    
    #diam = 50
    p = 0.443
    dx = p/16
    h = 0.6
    Np = 16
    xgrid = np.arange(-diam/2-1,diam/2+1,dx)
    ygrid = np.linspace(-p*3,p*3, Np*6)
    
    xx,yy = np.meshgrid(xgrid,ygrid)
    NPML = [20,20]

    eps = mesh(radData, xData, diam)
    sim = Simulation(omega, eps, dx, NPML, "Ez", use_dirichlet_bcs=True)
    sim.src[25,:] = 10
    sim.solve_fields()
    
    return sim.fields['Ez']

def computeLensEfficiency(rads, xloc, F, diam, zmax, dz):
    E = simulateLens(rads, xloc, F, diam)
    #zmax = 2000 # microns
    
    dx = 0.443/16
    #dz = dx
    wave = 0.633
    omega = 2*np.pi/wave
    
    #diam = 50
    
    xgrid = np.arange(-diam/2-5,diam/2+5,dx)
     
    zlist = np.arange(0,zmax,dz)
    Et = np.conj(E[63,:])
    Ef = np.zeros([len(Et),len(zlist)],dtype=np.complex)
    
    for i in range(0,len(zlist)):
        pr = Propagator1DPadded(len(Et), omega, zlist[i], dx, pad_factor=3., device = torch.device('cpu'))
        fieldAtFocalPlane = pr.prop(torch.tensor(Et))
        Ef[:,i] = fieldAtFocalPlane.numpy()
        
    I = np.abs(Ef)**2
    imax, jmax = np.unravel_index(np.argmax(I),I.shape)
    
    parameters, covariance = curve_fit(gauss, xgrid, I[:,jmax])
    FWHM = np.abs(2*np.sqrt(2*np.log(2))*parameters[3])

    fit_y = gauss(xgrid, parameters[0], parameters[1], parameters[2], parameters[3])
    
    
    filter = np.zeros(xgrid.shape)
    filter[(xgrid > -3*FWHM+parameters[2]) & (xgrid < 3*FWHM+parameters[2])] = 1
    If = filter*I[:,jmax]
    
    plt.figure()
    plt.plot(If)
    plt.plot(filter)
    eff = np.sum(If)/np.sum(np.abs(E[25,:])**2)
    
    dict = {"efficiency" : eff,
            "I": I,
            "fwhm": FWHM,
            "H": parameters[0],
            "A": parameters[1],
            "x0": parameters[2],
            "sigma": parameters[3],
            "xgrid" : xgrid,
            "flen": zlist[jmax]
            }
    return dict

#%% input Data

xData, radData = np.loadtxt('/home/noise/code/pinn/inv-design-2d/pinn-metagrating-optimization/lensOptimization/designedLensesSept13/fwd-50nmGap-f100.dat')#np.loadtxt('fwd-75nmGap-f500.dat')

Ezfdfd = simMS(xData,radData, 100)

plt.figure()
plt.imshow(np.abs(Ezfdfd)**2)
plt.colorbar()
plt.title(str(np.max(np.abs(Ezfdfd)**2)))
plt.show()
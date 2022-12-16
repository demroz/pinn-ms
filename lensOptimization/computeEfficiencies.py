from angler import Simulation
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
from torchPhysicsUtils import *
import torch

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
    Et = np.conj(E[65,:])
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

def computeEfficiencyFromField(Et,diam):
    zmax = 2200 # microns
    
    dx = 0.443/16
    dz = dx*50 
    wave = 0.633
    omega = 2*np.pi/wave
    
    #diam = 50
    
    xgrid = np.arange(-diam/2-5,diam/2+5,dx)
     
    zlist = np.arange(0,zmax,dz)
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
    eff = np.sum(If)/len(If)
    
    dict = {"efficiency" : eff,
            "I": I,
            "fwhm": FWHM,
            "H": parameters[0],
            "A": parameters[1],
            "x0": parameters[2],
            "sigma": parameters[3],
            "xgrid" : xgrid,
            "flen": zlist[jmax],
            "zlist": zlist
            }
    return dict
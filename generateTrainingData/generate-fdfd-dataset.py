#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:21:25 2022

@author: noise
"""
import os
import torch
import numpy as np
from angler import Simulation
import matplotlib.pyplot as plt 
import pandas as pd

def generateEps(rads, index, xx, yy, radlocs, h):
    eps = np.ones(xx.shape)
    for i in range(0,len(radlocs)):
        eps[(np.abs(xx-radlocs[i]) < rads[i]) &
            (np.abs(yy) < h/2)
            ] = index**2
    return eps

def generateDataset(parameterDictionary):
    '''
    Parameters
    ----------
    parameterDictionary : dictionary
        dictionary with simulation parameters
        
        Dataset Parameters
        'datasize' = (int) number of datapoints
        'numRadii' = (int) number of pillar radii to simulate
        
        Simulation parameters
        'index' = (float) refractive index
        'wave' = (float) wavelength in microns
        'dx' = (float) simulation resolution microns/pixel
        'p' = (flaot) periodicity
        'NPML' = (list of 2 values [pmlpix, pmlpix])
                    number of PML pixels in x,y direction
        'pol' = (string) polarization 'Ez' or 'Hz'
        'BC' = (bool) = True - dirichlet BC / False - periodic BCs
        
        Output Parameters
        'datadir' = (str) where to save dictionary
        'filename' = (str) name of dataset

    Returns
    -------
    dataset dictionary used to train neural network

    '''
    datasize = parameterDictionary['datasize']
    numRadii = parameterDictionary['numRadii']
    index = parameterDictionary['index']
    wave = parameterDictionary['wave']
    dx = parameterDictionary['dx']
    p = parameterDictionary['p']
    h = parameterDictionary['h']
    NPML = parameterDictionary['NPML']
    pol = parameterDictionary['pol']
    BC = parameterDictionary['BC']
    datadir = parameterDictionary['datadir']
    filename = parameterDictionary['filename']
    omega = 2*np.pi/wave
    
    dict = []
    for i in range(datasize):
        d = []
        rads = np.random.uniform(size=numRadii)*p/2
        
        Np = int(p/dx)
        x = np.linspace(-p*(len(rads)+1)/2,p*(len(rads)+1)/2,Np*(len(rads)+1))
        y = np.linspace(-p*3,p*3,Np*6)
        xx,yy = np.meshgrid(x,y)
        
        radlocs = np.arange(-np.floor(len(rads)/2), np.ceil(len(rads)/2))*p
        eps = generateEps(rads, index, xx, yy, radlocs, h)
        
        sim = Simulation(omega, eps, dx, NPML, pol)
        sim.src[NPML[0]+15,:] = 20
        sim.solve_fields()
        
        Ex_re = np.real(sim.fields['Ex'])
        Ex_im = np.imag(sim.fields['Ex'])
        
        Ey_re = np.real(sim.fields['Ey'])
        Ey_im = np.imag(sim.fields['Ey'])
        
        Ez_re = np.real(sim.fields['Ez'])
        Ez_im = np.imag(sim.fields['Ez'])
        
        Hx_re = np.real(sim.fields['Hx'])
        Hx_im = np.imag(sim.fields['Hx'])
        
        Hy_re = np.real(sim.fields['Hy'])
        Hy_im = np.imag(sim.fields['Hy'])
        
        Hz_re = np.real(sim.fields['Hz'])
        Hz_im = np.imag(sim.fields['Hz'])
        
        J = sim.src
        Dyb = sim.derivs['Dyb']
        Dyf = sim.derivs['Dyf']
        Dxb = sim.derivs['Dxb']
        Dxf = sim.derivs['Dxf']
        
        cc = Dxf.dot(Dxb)+Dyf.dot(Dyb)
        d = {
          'radii' : rads,
          'eps' : eps,
          'Exre' : Ex_re,
          'Exim' : Ex_im,
          'Eyre' : Ey_re,
          'Eyim' : Ey_im,
          'Ezre' : Ez_re,
          'Ezim' : Ez_im,
          'Hxre' : Hx_re,
          'Hxim' : Hx_im,
          'Hyre' : Hy_re,
          'Hyim' : Hy_im,
          'Hzre' : Hz_re,
          'Hzim' : Hz_im,
          'J' : J,
          'cc' : cc,
          'dxe': Dxb,
          'dxh': Dxf,
          'dye': Dyb,
          'dyh': Dyf
          }
        dict.append(d)
        
    df = pd.DataFrame.from_dict(dict)
    if not os.path.isdir(datadir):
        os.makedirs(datadir)
        
    df.to_pickle(datadir+filename)
        
#%%
parameterDictionary = {'datasize': 10000,
                       'numRadii': 11,
                       'index': 2,
                       'wave': 0.633,
                       'dx': 0.443/16,
                       'p': 0.443,
                       'h': 0.6,
                       'NPML': [10,0],
                       'pol': 'Ez',
                       'BC' : False,
                       'datadir': '/home/noise/code/pinn/inv-design-2d/pinn-metagrating-optimization/generateTrainingData/fdfd-data/',
                       'filename': 'fdfd-data.dat'}
generateDataset(parameterDictionary)

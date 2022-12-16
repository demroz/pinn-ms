# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:41:31 2022

@author: noise
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchPhysicsUtils import Propagator1D, Propagator1DPadded

wave = 0.633
omega = 2*np.pi/wave
period = 0.443

def lossEDOF(field, f0, f1):
    '''
    Parameters
    edof figure of merit
    '''
    zlist = np.linspace(f0, f1, 50)
    loss = []
    mid = int(len(field)/2)
    for z in zlist:
        p = Propagator1DPadded(len(field), omega, z, period, pad_factor=1., device = torch.device('cuda'))
        ff = p.prop(field)
        loss.append(-torch.abs(ff[mid])**2)
        
    zt = np.linspace(0,f1+50,101)
    I = np.zeros([len(field),len(zt)])
    i = 0
    for z in zt:
        p = Propagator1DPadded(len(field), omega, z, period, pad_factor=1., device = torch.device('cuda'))
        ff = p.prop(field)
        I[:,i] = torch.abs(ff).cpu().detach()**2
        i += 1
        
    plt.figure()
    plt.imshow(I)
    plt.gca().set_aspect(0.1)
    plt.show()
    return torch.max(torch.stack(loss))

def optimizePhase(initPhase,f0,f1):
    '''
    Parameters
    ----------
    initPhase : numpy arrayu
        initial condition
    f0 : float
        min focal length
    f1 : float
        max focal length

    Returns
    -------
    phase : 
        optimal phase

    '''
    phase = torch.tensor(initPhase.copy(), requires_grad = True, device="cuda")
    epochs = 100
    optimizer = torch.optim.Adam([phase], lr=0.1)
    train_loss_ar = []
    for epoch in range(epochs):
        with torch.no_grad():
            phase.clamp_(0,2*np.pi)   

        optimizer.zero_grad()
        
        field = torch.exp(1j*phase)
        l = lossEDOF(field, f0, f1)
        l.backward()
        optimizer.step()
        train_loss_ar.append(l.item())
        print(epoch, "out of", epochs)
        
        if epoch%50==0:
            plt.figure()
            plt.plot(train_loss_ar)
            plt.show()
            
    return phase.detach().cpu().numpy()

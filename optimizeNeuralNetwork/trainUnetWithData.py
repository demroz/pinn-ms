import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt 
from angler import Simulation
import pandas as pd
from UNetArch import *
from plotUtil import *

#%%
ALPHA = 10
#%% test dataset
datafile = '/home/noise/code/pinn/inv-design-2d/pinn-metagrating-optimization/generateTrainingData/fdfd-data/fdfd-data.dat'
data = pd.read_pickle(datafile)
#%%
train_data = data.loc[0:9000]
test_data = data.loc[9000:10000] # only test 100 samples from full dataset

Nx = train_data['Ezre'][0].shape[0] # field shape
Ny = train_data['Ezre'][0].shape[1]

# double curl op
cct = torch.tensor(data['cc'][0].todense(),dtype=torch.complex64).cuda()
#%% physical parameters 
Np = 16
p = 0.443
h = 0.6
dx = p/Np
wave = 0.633
omega = 2*np.pi/wave
J = 1j*omega*torch.tensor(data['J'][0].reshape(1,Nx*Ny)).cuda() # copy current source

#%% neural net definition
nnre = UNet(1)
nnim = UNet(1)
nnre.cuda()
nnim.cuda()

#%% output data locations
savedir = "nnTrainingData/"
slicesdir = savedir+"slices/"
fielddir = savedir+"fields/"
pdfdir = savedir+"pdf/"

#%%
def nnLoss(E,F,eps, Ebar, Fbar):
    '''
    Neural network loss function

    Parameters
    ----------
    E : torch tensor
        real part of NN predicted E field
    F : torch tensor
        imag part --//--
    eps : torch tensor
        discretized epsilon field

    Returns
    -------
    tensor
        ||\nabla^2 E + omega^2 epsilon E - J||^2

    '''
    alpha= 2
    eps = eps.reshape(-1,Nx*Ny)
    field = E+1j*F
    l1 = cct@field.T+omega**2*eps.T*field.T-J.T
    print("loss",torch.mean(torch.abs(l1)))
    l2 = torch.abs(field-(Ebar+1j*Fbar))
    print("loss2",torch.mean(torch.abs(l2)))
    
    return torch.mean(torch.abs(l1))+alpha*torch.mean(l2), torch.mean(torch.abs(l1)), torch.mean(l2)

def testDataLoss(E, F, Etar_re, Etar_im, eps):
    '''
    
    producted loss statistics on test data
    
    Parameters
    ----------
    E : nn predicted real field
    F : nn predicted imagninary field
    
    Etar_re : target real field
    Etar_im : target imagninary field
    eps : epsilon field

    Returns
    -------
    None.

    '''
    nnfield = E+1j*F
    target_field = Etar_re+1j*Etar_im
    eps = eps.reshape(-1,Nx*Ny)
    
    l1 = cct@nnfield.T+omega**2*eps.T*nnfield.T-J.T
    l2 = torch.sum(torch.abs(nnfield-target_field))
    
    print("pinn loss", l1.item(), "l1 loss", l2.item())
    
def getTestBatch(numeps, epoch):
    '''
    Parameters
    ----------
    numeps : int
        number of epsilon fields
    epoch : int
        epoch #

    Returns
    -------
    eps : tensor
        test epsilon field
    Etar_re : tensor
        real part of Ez
    Etar_im : tensor
        complex part of Ez

    '''
    w = 9000
    inds = np.arange(epoch*numeps, (epoch+1)*numeps)
    eps = np.zeros((numeps,1,Nx,Ny))
    Etar_re = np.zeros((numeps,Nx,Ny))
    Etar_im = np.zeros((numeps,Nx,Ny))
    for i in range(0,numeps):
        eps[i,0,:,:] = test_data['eps'][inds[i]+w]
        Etar_re[i,:,:] = test_data['Ezre'][inds[i]+w]
        Etar_im[i,:,:]  = test_data['Ezim'][inds[i]+w]
        
    eps = torch.tensor(eps,dtype=torch.float32)
    return eps, Etar_re, Etar_im

def generateEps(r):
    '''
    Parameters
    ----------
    r : tensor
        list of radii

    Returns
    -------
    eps : tensor
        epsilon field

    '''
    x = np.linspace(-p*12/2,p*12/2,Np*12)
    y = np.linspace(-p*3,p*3,Np*6)
    xx,yy = np.meshgrid(x,y)
    radlocs = np.arange(-5,6)*p
    eps = np.ones(xx.shape)
    for i in range(0,len(radlocs)):
        eps[(np.abs(xx-radlocs[i]) < r[i]) &
            (np.abs(yy) < h/2)
            ] = 2**2
    return eps

def simulateFDFD(eps):
    numeps = len(eps)
    NPML = [10,0]
    
    E = torch.zeros(numeps,Nx,Ny)
    F = torch.zeros(numeps,Nx,Ny)
    for i in range(numeps):
        eps_sim = eps[i,0,:,:].detach().cpu().numpy()
        sim = Simulation(omega, eps_sim, dx, NPML, 'Ez')
        sim.src[NPML[0]+15,:] = 20
        sim.solve_fields()
        E[i,:,:] = torch.tensor(np.real(sim.fields['Ez'])).cuda()
        F[i,:,:] = torch.tensor(np.imag(sim.fields['Ez'])).cuda()
    
    return E,F

def getTrainBatch(batchsize):
    inds = np.random.randint(0,9000,batchsize)
    eps = np.zeros((batchsize,1,Nx,Ny))
    Etar_re = np.zeros((batchsize,Nx,Ny))
    Etar_im = np.zeros((batchsize,Nx,Ny))
    for i in range(0,batchsize):
        eps[i,:,:,:] = train_data['eps'][inds[i]]
        Etar_re[i,:,:] = train_data['Ezre'][inds[i]]
        Etar_im[i,:,:]  = train_data['Ezim'][inds[i]]
        
    eps = torch.tensor(eps,dtype=torch.float32)
    return eps, Etar_re, Etar_im


q= getTrainBatch(10)

#%% training loop
nepochs = 50000
params = list(nnre.parameters()) + list(nnim.parameters())
pp = 1e-4 # optimizer training rate
optimizer = torch.optim.Adam(params, lr=pp, weight_decay=1e-5)

train_loss_ar = [] # training loss
test_loss_ar = [] # testing loss
test_norm_error_ar = []
epochNo = [] # for testing

resloss = []
test_res_loss = []
err_loss = []
test_err_loss = []

for epoch in range(nepochs):
    
    batch_eps, Ebar, Fbar = getTrainBatch(10)
    batch_eps =  batch_eps.cuda()
    Ebar = torch.tensor(Ebar)
    Fbar = torch.tensor(Fbar)
    # loss
    optimizer.zero_grad()
    Ebar = Ebar.reshape(10,Nx*Ny).cuda()
    Fbar = Fbar.reshape(10,Nx*Ny).cuda()
    
    loss, l1, l2 = nnLoss(nnre(batch_eps),nnim(batch_eps),batch_eps, Ebar, Fbar)
    loss.backward()
    optimizer.step()
    train_loss_ar.append(loss.item()) # loss data
    resloss.append(l1.item())
    err_loss.append(l2.item())
    # trainig visualization
    print(epoch)
    if epoch%1000 == 0:
        epochNo.append(epoch)
        nnre.cpu()
        nnim.cpu()
        
        trainBatchSize = 10
        nBatches = int(len(test_data)/trainBatchSize)
        q = 0
        
        test_batch_loss = []
        test_batch_norm_error = []
        tl1_q = 0
        tl2_q = 0
        for testEpoch in range(nBatches):
            epst, Etest, Ftest = getTestBatch(trainBatchSize, testEpoch)
            Enn = nnre(epst).reshape(trainBatchSize, Nx, Ny)
            Fnn = nnim(epst).reshape(trainBatchSize, Nx, Ny)
            
            testLoss, tl1, tl2 = nnLoss(nnre(epst).cuda(), nnim(epst).cuda(),epst.cuda(), torch.tensor(Etest.reshape(10,Nx*Ny)).cuda(), torch.tensor(Ftest.reshape(10,Nx*Ny)).cuda())
            test_batch_loss.append(testLoss.item())
            
            nnfield = Enn+1j*Fnn
            tfield = Etest+1j*Ftest
            
            norm_error = np.sum(np.abs(nnfield.detach().numpy()-tfield)**2)/np.sum(np.abs(tfield)**2)
            test_batch_norm_error.append(norm_error)
            tl1_q += tl1.item()/nBatches
            tl2_q += tl2.item()/nBatches
        test_res_loss.append(tl1_q)
        test_err_loss.append(tl2_q)
            #for j in range(trainBatchSize):
            #    plotFullFields(nnfield[j,:,:], tfield[j,:,:], epoch, q, fielddir)
           #     q +=1

        test_loss_ar.append(np.mean(test_batch_loss))
        test_norm_error_ar.append(np.mean(test_batch_norm_error))
        
        plt.figure()
        plt.plot(np.log(train_loss_ar))
        plt.plot(epochNo,np.log(test_loss_ar))
        plt.legend(['train loss', 'test loss'])
        plt.show()
        
        plt.figure()
        plt.plot(test_norm_error_ar)
        plt.legend(['F norm error'])
        plt.show()
        
        plt.figure()
        plt.plot(np.log(resloss))
        plt.plot(epochNo,np.log(test_res_loss))
        plt.legend(['res','test res'])
        plt.show()
        
        plt.figure()
        plt.plot(np.log(err_loss))
        plt.plot(epochNo,np.log(test_err_loss))
        plt.legend(['res','test res'])
        plt.show()
        
        nnre.cuda()
        nnim.cuda()
     
np.savetxt(savedir+'train_loss'+str(ALPHA), train_loss_ar)
np.savetxt(savedir+'test_loss'+str(ALPHA), test_loss_ar)
np.savetxt(savedir+'fro_loss'+str(ALPHA), test_norm_error_ar)
np.savetxt(savedir+'train_res_loss'+str(ALPHA), resloss)
np.savetxt(savedir+'test_res_loss'+str(ALPHA), test_res_loss)
np.savetxt(savedir+'err_loss'+str(ALPHA), err_loss)
np.savetxt(savedir+'test_err_loss'+str(ALPHA), test_err_loss)

#%%

import scipy.io as sio

dict = {}
dict["trainLoss"] = train_loss_ar
dict["testLoss"] = test_loss_ar
dict["froLoss"] = test_norm_error_ar
dict["train_res"] = resloss
dict["test_res"] = test_res_loss
dict["train_mae"] = err_loss
dict["test_mae"] = test_err_loss

sio.savemat(savedir+"lossStatsNN_alpha"+str(ALPHA)+".mat",dict)
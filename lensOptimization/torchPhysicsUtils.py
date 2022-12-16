"""
Created on Mon Apr 25 09:53:24 2022

@author: noise
"""
import torch
import numpy as np
import time

class Propagator1D:
    @staticmethod
    def _get_grid(nx, dx):
        x = (np.arange(nx) - (nx-1.) / 2) * dx
        return x
    
    @staticmethod
    def _get_frequencies(nx, dx):
        k_x = np.fft.fftfreq(n=nx, d=dx) * 2 * np.pi
        return k_x
    
    @staticmethod
    def _setup_H_tensor(nx, k, z_list, dx, device):
        k_x = np.fft.fftfreq(n=nx, d = dx) * 2 * np.pi
        #z_list = np.array(z_list)#torch.tensor(z_list, device = cuda)
        k_z = np.sqrt(0j + k ** 2 - k_x ** 2)
        k_z = torch.tensor(k_z, device = device)
        
        phase = k_z * z_list
        
        H = torch.exp(1j * phase)
        
        return H
    
    def __init__(self, nx, k, z_list, dx, device):
        self.H = Propagator1D._setup_H_tensor(nx, k, z_list, dx, device)
        self.nx = nx
        self.dx = dx
        
    def prop(self, field):
        U = torch.fft.fft(field)
        
        E_k_prop = U * self.H
        E_prop = torch.fft.ifft(E_k_prop)
        
        return E_prop
    
class Propagator1DPadded(Propagator1D):
    def __init__(self, nx, k, z_list, dx, pad_factor = 1., device = torch.device('cuda')):
        self.nx = nx
        self.dx = dx
        self.x = Propagator1D._get_grid(nx, dx)
        
        self.pad_factor = pad_factor
        
        nx_padded = nx + 2 * int(nx * pad_factor / 2)
        self.propagator = Propagator1D(nx_padded, k, z_list, dx, device)
        
    @staticmethod
    def _pad(source, pad_factor = 1.):
        n_x = len(source)
        pad_x = int(n_x * pad_factor / 2)
        return torch.nn.functional._pad(input=source, pad=(pad_x, pad_x), mode='constant', value=0)

    def _unpad(source, pad_factor = 1.):
        if pad_factor == 0.:
            return source
        else:
            n_x = len(source)
            pad_x = int(n_x * pad_factor / (2 + 2 * pad_factor))
            return source[pad_x:-pad_x]
        
    def prop(self, field):
        field = Propagator1DPadded._pad(field, pad_factor = self.pad_factor)
        field = self.propagator.prop(field)
        field = Propagator1DPadded._unpad(field, pad_factor = self.pad_factor)
        
        return field
    
class RayleighSommerfieldPropagation():
    def __init__(self, k, dx, device = torch.device('cpu')):
        self.dx = dx
        self.k = k
        self.device = device
        
    def propagate(self, U, xp, x, z):
        '''
        Parameters
        ----------
        U : tensor
            nearfield at U(x',0)
        xp : tensor
            nearfield coordinate
        x : tensor
            farfield coordinate x
        z : tensor
            farfield coordinate z

        Returns
        -------
        farfied at x, z

        '''
        r = torch.sqrt((x-xp)**2+z**2)
        H = z*torch.exp(1j*self.k*r)/r
        
        farfield = torch.sum(1.0/(1j*2*np.pi/self.k)*U*H*self.dx)
        
        return farfield
# Turbulent velocity generator across a CC

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import multiprocessing
import warnings
import time
from itertools import product
import pickle
import scipy.fft as fft

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

def velocity_generator(n=[100,100,100], kmin=1, kmax=50, beta=1./3., vmax=30.):
   np.random.seed(100)
   alpha = 2.*beta + 1.
   kx = np.zeros(n)
   ky = np.zeros(n)
   kz = np.zeros(n)

   for j in range(0,n[1]):
     for k in range(0,n[2]):
        kx[:,j,k] = n[0]*fft.fftfreq(n[0])
   for i in range(0,n[0]):
     for k in range(0,n[2]):
        ky[i,:,k] = n[1]*fft.fftfreq(n[1])       
   for i in range(0,n[0]):
     for j in range(0,n[1]):
        kz[i,j,:] = n[2]*fft.fftfreq(n[2]) 

   kx = np.array(kx)
   ky = np.array(ky)
   kz = np.array(kz)
   k = np.sqrt(kx**2. + ky**2. + kz**2.)

   indx = np.where(np.logical_and(k**2. >= kmin**2., k**2. < (kmax+1)**2.))
   nr = len(indx[0])

   phasex = np.zeros(n)
   phasex[indx] = 2.*np.pi*np.random.uniform(size=nr)
   fx = np.zeros(n)
   fx[indx] = np.random.normal(size=nr)

   phasey = np.zeros(n)
   phasey[indx] = 2.*np.pi*np.random.uniform(size=nr)
   fy = np.zeros(n)
   fy[indx] = np.random.normal(size=nr)

   phasez = np.zeros(n)
   phasez[indx] = 2.*np.pi*np.random.uniform(size=nr)
   fz = np.zeros(n)
   fz[indx] = np.random.normal(size=nr)

   for i in range(int(kmin), int(kmax+1)):
      indx_slice = np.where(np.logical_and(k>=i, k<i+1))
      rescale = np.sqrt(np.sum(np.abs(fx[indx_slice])**2. + np.abs(fy[indx_slice])**2. + np.abs(fz[indx_slice])**2.))
      fx[indx_slice] = fx[indx_slice]/rescale
      fy[indx_slice] = fy[indx_slice]/rescale
      fz[indx_slice] = fz[indx_slice]/rescale
   
   fx[indx] = fx[indx]*k[indx]**(-0.5*alpha)
   fy[indx] = fy[indx]*k[indx]**(-0.5*alpha)
   fz[indx] = fz[indx]*k[indx]**(-0.5*alpha)

   fx = np.cos(phasex)*fx + 1j*np.sin(phasex)*fx
   fy = np.cos(phasey)*fy + 1j*np.sin(phasey)*fy
   fz = np.cos(phasez)*fz + 1j*np.sin(phasez)*fz

   pertx = np.real(np.fft.ifftn(fx))
   perty = np.real(np.fft.ifftn(fy))
   pertz = np.real(np.fft.ifftn(fz))
   
   pertx = pertx - np.average(pertx)
   perty = perty - np.average(perty)
   pertz = pertz - np.average(pertz)
   
   norm = np.sqrt(np.sum(pertx**2. + perty**2. + pertz**2.)/np.product(n))
   pertx = pertx/norm
   perty = perty/norm
   pertz = pertz/norm
   
   pertx = vmax * pertx / np.std(pertx) / np.sqrt(3.)
   perty = vmax * perty / np.std(perty) / np.sqrt(3.)
   pertz = vmax * pertz / np.std(pertz) / np.sqrt(3.)
  
   return pertx, perty, pertz, k


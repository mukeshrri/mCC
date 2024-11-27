# Generating cloudlets
# Next:- go to cloud_analysis.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from decimal import Decimal
import multiprocessing
import warnings
import time
from itertools import product
import pickle
from tqdm import tqdm
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()
    
mu = 0.6
mp = 1.6726e-24
Msun = 1.989e33
pc = 3.086e18

distri = 'u'

def power_law(a_l, a_u, a0, alpha, size):    
    cpdf_samples = np.random.uniform(low=0.0, high=1.0, size=size)
    return a0*((a_l/a0)**(1-alpha)*(1-cpdf_samples) + cpdf_samples*(a_u/a0)**(1-alpha))**(1/(1-alpha))

def cloud_gen(n_cold, 
              Rcc,
              rc,
              Mcc):   

   print('n_cold:', n_cold,
         'Rcc:', Rcc,
         'rc:', rc,
         'Mcc:', Mcc)
   
   Nc = Mcc*Msun/(4./3.*np.pi*(rc*pc)**3.*n_cold*mu*mp)
   print('No of cloudlets are : ', Nc)
   print('area covering fraction is : ', Nc*(rc/Rcc/1.e3)**2.)
   print('volumne fraction is : ', Nc*(rc/Rcc/1.e3)**3.)
         
   # generate cloudlets
   X = np.zeros(int(Nc))
   Y = np.zeros(int(Nc))
   Z = np.zeros(int(Nc))
   
   if distri == 'u':
     for i in range(int(Nc)):
      X[i] = np.random.uniform(low=-Rcc, high=Rcc, size=1)[0]
      Y[i] = np.random.uniform(low=-Rcc, high=Rcc, size=1)[0]
      Z[i] = np.random.uniform(low=-Rcc, high=Rcc, size=1)[0]
      dist = np.sqrt(X[i]**2. + Y[i]**2. + Z[i]**2.) 
      while dist>Rcc:
        X[i] = np.random.uniform(low=-Rcc, high=Rcc, size=1)[0]
        Y[i] = np.random.uniform(low=-Rcc, high=Rcc, size=1)[0]
        Z[i] = np.random.uniform(low=-Rcc, high=Rcc, size=1)[0]
        dist = np.sqrt(X[i]**2. + Y[i]**2. + Z[i]**2.) 
   
   if distri == 'pl':
   
     r = power_law(D_min, D_max, 10.0, alpha, int(Nc))
     phi = 2.*np.pi*np.random.uniform(low=0.0, high=1.0, size=int(Nc))
     theta = np.arccos(2*np.random.uniform(low=0.0, high=1.0, size=int(Nc)) -1.)
     X0 = r*np.sin(theta)*np.cos(phi) 
     Y0 = r*np.sin(theta)*np.sin(phi) 
     Z0 = r*np.cos(theta) 
   
     X = X0.astype(dtype=np.float16)
     Y = Y0.astype(dtype=np.float16)
     Z = Z0.astype(dtype=np.float16)
   
   fl = h5py.File("./data/data_{}_{}_{}_{}.hdf5".format(n_cold,Rcc,rc,np.log10(Mcc)), "a")
   fl.create_dataset('X', data = X)
   fl.create_dataset('Y', data = Y)
   fl.create_dataset('Z', data = Z)
   fl.create_dataset('info', data = np.array([Mcc,n_cold,Rcc,rc,Nc]))
   fl.close()  
   
   print('complete! \n')
   
if __name__ == "__main__":
    n_cold_ = [0.01] # cm^-3
    Rcc_ = [15] # kpc
    rc_ = [60,40,20,15,10] # pc
    Mcc_ = [1e7] # solar mass

    for condition in product(n_cold_, 
                             Rcc_,
                             rc_,
                             Mcc_):
                             
        cloud_gen(*condition)
            
        print('complete! \n')
        print("--- %s seconds ---\n " % (time.time() - start_time)) 

        

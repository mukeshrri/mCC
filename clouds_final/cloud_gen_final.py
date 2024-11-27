# generate ellipsoidal cloudlets 
# Next:- go to cloud_analysis.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from decimal import Decimal
from scipy.spatial.transform import Rotation as rot
from scipy.spatial.distance import squareform,pdist
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
kpc = 3.086e21
fac = mu*mp*kpc**3/Msun

N_los = 1.e4

def power_law(a_l, a_u, a0, alpha, size):    
    cpdf_samples = np.random.uniform(low=0.0, high=1.0, size=size)
    return a0*((a_l/a0)**(1-alpha)*(1-cpdf_samples) + cpdf_samples*(a_u/a0)**(1-alpha))**(1/(1-alpha))

def cloud_gen(N_cc, 
              M_cold, 
              n_cold, 
              rCGM, 
              D_min, 
              D_max, 
              alpha,
              d_min,
              d_max,
              beta,
              a_min,
              a_max,
              gamma):   

   print('N_cc:', N_cc,
         'M_cold:', np.log10(M_cold), 
         'n_cold:', n_cold, 
         'rCGM:', rCGM,
         'D_min:', D_min, 
         'D_max:', D_max, 
         'alpha:', alpha,
         'd_min:', d_min,
         'd_max:', d_max,
         'beta:', beta,
         'a_min:', a_min,
         'a_max:', a_max,
         'gamma:', gamma)
   
   MCLD = int(np.log10(M_cold)) if M_cold != 5.e9 else 5.9
         
   # generate cloud complex
   r = power_law(D_min, D_max, 10.0, alpha, N_cc)
   phi = 2.*np.pi*np.random.uniform(low=0.0, high=1.0, size=N_cc)
   theta = np.arccos(2*np.random.uniform(low=0.0, high=1.0, size=N_cc) -1.)
   X0 = r*np.sin(theta)*np.cos(phi) 
   Y0 = r*np.sin(theta)*np.sin(phi) 
   Z0 = r*np.cos(theta) 
   
   X0 = X0.astype(dtype=np.float16)
   Y0 = Y0.astype(dtype=np.float16)
   Z0 = Z0.astype(dtype=np.float16)

   # generate cloudlets
   A = []  # x semi-axis length
   B = []  # y semi-axis length
   C = []  # z semi-axis length
   x0 = [] # x coordinate of cloud center
   y0 = [] # y coordinate of cloud center
   z0 = [] # z coordinate of cloud center

   vol = 0
   mass = 0
   q = 0  # cloud/cloudlet count
   p = 0  # complex/cluster count
   while mass<M_cold:
        
        aspect = power_law(a_min, a_max, 1.0, gamma, 3)
        a = aspect[0]
        b = aspect[1]
        c = aspect[2]
        A.append(a)
        B.append(b)
        C.append(c)
    
        vol += a*b*c
        r = power_law(d_min, d_max, 1.0, beta, 1)[0]
        phi = 2.*np.pi*np.random.uniform(low=0.0, high=1.0, size=1)[0]
        theta = np.arccos(2*np.random.uniform(low=0.0, high=1.0, size=1)[0] -1.)
        xc = r*np.sin(theta)*np.cos(phi) + X0[p]
        yc = r*np.sin(theta)*np.sin(phi) + Y0[p]
        zc = r*np.cos(theta) + Z0[p]
        dist = np.sqrt(xc**2. + yc**2. + zc**2.) + max(a,b,c)
        while dist>rCGM:
            r = power_law(d_min, d_max, 1.0, beta, 1)[0]
            phi = 2.*np.pi*np.random.uniform(low=0.0, high=1.0, size=1)[0]
            theta = np.arccos(2*np.random.uniform(low=0.0, high=1.0, size=1)[0] -1.)
            xc = r*np.sin(theta)*np.cos(phi) + X0[p]
            yc = r*np.sin(theta)*np.sin(phi) + Y0[p]
            zc = r*np.cos(theta) + Z0[p]
            dist = np.sqrt(xc**2. + yc**2. + zc**2.) + max(a,b,c)
        x0.append(xc)   
        y0.append(yc)
        z0.append(zc)
        mass += n_cold*4./3.*np.pi*a*b*c*fac    
        p = (p + 1)*(1 if p+1<N_cc else 0)
        q += 1
        
        if q == int(1.e3) or q == int(1.e4) or q == int(1.e5) or q == int(1.e6) or q == int(1.e7) or q == int(1.e8) or q == int(1.e9) or q == int(1.e10):
           print(np.log10(q))
  
   N_cl = q # total number of clouds
   f_V = vol/rCGM**3.

   print('Number of cloud complex are :', N_cc)
   print('Number of clouds/cloudlets are :', N_cl)
   print('Initial cold mass is (M_sun): {:.1e}'.format(M_cold))
   print('Total cold mass is (M_sun): {:.3e}'.format(mass))
   print('volume fraction is : {:.1e}'.format(f_V))
 
   angl_alfa = np.random.uniform(low=0, high=2*np.pi, size=N_cl)
   angl_beta = np.random.uniform(low=0, high=2*np.pi, size=N_cl)
   angl_gmma = np.random.uniform(low=0, high=2*np.pi, size=N_cl)
   
   angl_alfa = angl_alfa.astype(dtype=np.float16)
   angl_beta = angl_beta.astype(dtype=np.float16)
   angl_gmma = angl_gmma.astype(dtype=np.float16)
   
   A = np.array(A, dtype=np.float16)
   B = np.array(B, dtype=np.float16)
   C = np.array(C, dtype=np.float16)
   x0 = np.array(x0, dtype=np.float16)
   y0 = np.array(y0, dtype=np.float16)
   z0 = np.array(z0, dtype=np.float16)
   
   fl = h5py.File("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.hdf5".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), "a")
   fl.create_dataset('X0', data = X0)
   fl.create_dataset('Y0', data = Y0)
   fl.create_dataset('Z0', data = Z0)
   fl.create_dataset('x0', data = x0)
   fl.create_dataset('y0', data = y0)
   fl.create_dataset('z0', data = z0)
   fl.create_dataset('a', data = A)
   fl.create_dataset('b', data = B)
   fl.create_dataset('c', data = C)
   fl.create_dataset('angl_alfa', data = angl_alfa)
   fl.create_dataset('angl_beta', data = angl_beta)
   fl.create_dataset('angl_gmma', data = angl_gmma)
   fl.create_dataset('info', data = np.array([int(np.log10(M_cold)),np.log10(mass),n_cold,f_V,N_cc,N_cl,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma]))
   fl.close()  
   
   print('complete! \n')
   
if __name__ == "__main__":
    rCGM_ = [280]
    M_cold_ = [1.e10]
    n_cold_ = [0.01]
    N_cc_ = [1000]
    D_min_ = [10]
    D_max_ = [280]
    alpha_ = [1.2]
    d_min_ = [0.1]
    d_max_ = [20]
    beta_ = [0.2]
    a_min_ = [0.01]
    a_max_ = [1]
    gamma_ = [0.2]

    for condition in product(N_cc_, 
                             M_cold_, 
                             n_cold_, 
                             rCGM_, 
                             D_min_, 
                             D_max_, 
                             alpha_,
                             d_min_,
                             d_max_,
                             beta_,
                             a_min_,
                             a_max_,
                             gamma_):
                             
        cloud_gen(*condition)
            
        print('complete! \n')
        print("--- %s seconds ---\n " % (time.time() - start_time)) 


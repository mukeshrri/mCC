# calculate EW

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
import velocity_generator as turb_vel
import absorption_calculator as absorption

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

mu = 0.67
mp = 1.6726e-24
Msun = 1.989e33
kpc = 3.086e21
kms = 1.e5
kB = 1.3807e-16
amu = 1.6737e-24
c = 2.9979e10
fac = 4./3.*np.pi*mu*mp*kpc**3/Msun

M_star = 6.08e10
Temp = 1.e4
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
a_ion = 24.305 
mass_ion = a_ion*amu
b_dopp = np.sqrt(2*kB*Temp/mass_ion)/kms # doppler width in km/s
wave0 = 2796.3553 # rest-frame transition in Angstrom
spectral_res = 0.01 # spectral resolution in km/s

N_los = 1.e4

RCGM = 280

div = 700

vmax = np.sqrt(2.)*50.

pertx,perty,pertz,k_mode = turb_vel.velocity_generator(n=[div,div,div], kmin=1, kmax=100, beta=1./3., vmax=vmax)

X=Y=Z = np.linspace(-RCGM, RCGM, div)
   
def intersect_clouds():
    
    with h5py.File("./data/los_data_4.hdf5", 'r') as f:
         x_los = np.array(f['x_los'])
         y_los = np.array(f['y_los'])
    
    r_los = np.sqrt(x_los**2. + y_los**2.)
    no_sightlines = N_los
  
    cnt = np.zeros(r_los.shape[0])
    col_den = np.zeros(r_los.shape[0])
    EWidth = np.zeros(r_los.shape[0])
  
    Mg_density = 1.7e-6*n_cold
    
    for i in range(r_los.shape[0]):
   
       if_int = isinstance(res[i], int)
       if if_int:
          cnt[i] = res[i]
          continue
          
       if_tuple = isinstance(res[i], tuple)
       if if_tuple:
          cnt[i] = np.array(res[i][1]).shape[0]
          arr = np.array(res[i][0])
          arr1 = np.array(res[i][1])
          if np.isinf(arr).any() == True:
             print('problem at : ', i)
             print(arr)
             print(np.array(res[i][1]))
             sys.exit()
          
          col_ind = arr*Mg_density*kpc   
          col_den[i] = np.sum(col_ind)
       
          v_los = np.zeros(int(cnt[i]))
          for j in range(int(cnt[i])):
              x_indx = np.argmin(np.abs(X - x0[int(arr1[j])]))
              y_indx = np.argmin(np.abs(Y - y0[int(arr1[j])]))
              z_indx = np.argmin(np.abs(Z - z0[int(arr1[j])]))
              v_los[j] = pertz[x_indx,y_indx,z_indx]
       
          vel_all = np.linspace(min(v_los)-4.*b_dopp, max(v_los)+4.*b_dopp, 100)
       
          tau_total = np.zeros(vel_all.shape[0])
      
          for j in range(int(cnt[i])):
            N_l = col_ind[j]
            nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_dopp, wave0, v_los[j], spectral_res)
            norm_flux = absorption.generate_norm_flux(op_depth)
            wavel = c/nu * 1.e8
            vel_range = (wavel-wave0)/wave0 * c /kms
         
            tau_total += np.interp(vel_all, vel_range, op_depth)
          
          flux_total = np.exp(-tau_total) 
          wavel = wave0 * (1. + vel_all*kms/c)
      
          ew = np.trapz((1. - flux_total), wavel)
     
          EWidth[i] = ew 
       
    np.savetxt("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), np.column_stack((cnt,col_den,EWidth)), fmt='%.4e')
          
if __name__ == "__main__":
    rCGM_ = [280]
    M_cold_ = [5.e9]
    n_cold_ = [0.01]
    N_cc_ = [1000]
    D_min_ = [10]
    D_max_ = [280]
    alpha_ = [1.2]
    d_min_ = [0.1]
    d_max_ = [10]
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
                             
        N_cc = condition[0]
        M_cold = condition[1] 
        n_cold = condition[2] 
        rCGM = condition[3] 
        D_min = condition[4] 
        D_max = condition[5] 
        alpha = condition[6]
        d_min = condition[7]
        d_max = condition[8]
        beta = condition[9]
        a_min = condition[10]
        a_max = condition[11]
        gamma = condition[12]
         
        MCLD = int(np.log10(M_cold)) if M_cold != 5.e9 else 5.9
        
        print('N_cc:', N_cc,
         'M_cold:', MCLD, 
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
        
        with h5py.File("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.hdf5".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), 'r') as f:
            
             x0 = np.array(f['x0'])
             y0 = np.array(f['y0'])
             z0 = np.array(f['z0'])
             info = np.array(f['info'])
        
        x0 = np.array(x0, dtype=np.float32)
        y0 = np.array(y0, dtype=np.float32)
        z0 = np.array(z0, dtype=np.float32)
        
        N_cl = info[5]
        print(N_cl)
        
        fl = open("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pickle".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma),'rb')
        
        data_cl = pickle.load(fl)
        fl.close()  
   
        for key_val in data_cl.keys():
           exec(f"{key_val} = data_cl[\"{key_val}\"]")
        res = res
        
        intersect_clouds()
        
        print('complete! \n')
        
    print("--- %s seconds ---\n " % (time.time() - start_time)) 


        

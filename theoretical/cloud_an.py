# Getting the EW

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import multiprocessing
import warnings
import time
from itertools import product
import pickle
import absorption_calculator as absorption
from scipy import integrate
import velocity_generator as turb_vel
from scipy.optimize import minimize
import scipy.signal as signal

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

proc = 7

mu = 0.6
mp = 1.6726e-24
Msun = 1.989e33
kpc = 3.086e21
kB = 1.3807e-16
c = 2.9979e10
amu = 1.6737e-24
kms = 1.e5

rCGM = 280
M_star = 6.08e10
Nlos = 1.e4

mg_factor = 1.7e-6
Temp = 1.e4
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
wave0 = 2796.3553 # rest-frame transition in Angstrom
a_ion = 24.305 
mass_ion = a_ion*amu
b_dop = np.sqrt(2*kB*Temp/mass_ion)/kms # doppler width in km/s
spectral_res = 0.001 # spectral resolution in km/s

vmax = np.sqrt(2.)*50.
div = 600
pertx,perty,pertz,k_mode = turb_vel.velocity_generator(n=[div,div,div], kmin=1, kmax=100, beta=1./3., vmax=vmax)

X0=Y0=Z0 = np.linspace(-rCGM, rCGM, div)

print('vel gen complete!')

def get_EW(ress):
    
    if_tuple = isinstance(ress, tuple)
    if if_tuple:
       cnt = np.array(ress[0]).shape[0]
       arr = np.array(ress[1])
       arr1 = np.array(ress[0])
       if np.isinf(arr).any() == True:
          print('problem at : ')
          print(arr)
          print(np.array(ress[1]))
          sys.exit()
        
       col_den = arr
       
       v_los = np.zeros(int(cnt))
      
       for j in range(int(cnt)):
         x_indx = np.argmin(np.abs(X0 - X[int(arr1[j])]))
         y_indx = np.argmin(np.abs(Y0 - Y[int(arr1[j])]))
         z_indx = np.argmin(np.abs(Z0 - Z[int(arr1[j])]))
         v_los[j] = pertz[x_indx,y_indx,z_indx]
         
       vel_all = np.linspace(min(v_los)-4.*b_tot, max(v_los)+4.*b_tot, 100)
       tau_total = np.zeros(vel_all.shape[0])
       
       for j in range(int(cnt)):
         N_l = col_den[j]
         nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_tot, wave0, v_los[j], spectral_res)
         #nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_dop, wave0, v_los[j], spectral_res)
         norm_flux = absorption.generate_norm_flux(op_depth)
         wavel = c/nu * 1.e8
         vel_range = (wavel-wave0)/wave0 * c /kms
         
         tau_total += np.interp(vel_all, vel_range, op_depth)
          
       flux_total = np.exp(-tau_total) 
       wavel = wave0 * (1. + vel_all*kms/c)
       ew = np.trapz((1. - flux_total), wavel)
      
       return ew
    
    else :
       ew = 0
      
       return ew      
    
if __name__ == "__main__":

   Mcold_ = [1.e10]
   Ncc_ = [1.e3]
   Rcc_ = [10]
   
   for condition in product(Mcold_,Ncc_,Rcc_):
       
       Mcold = condition[0]
       Ncc = condition[1]
       Rcc = condition[2]
       Mcc = Mcold/Ncc
       
       b_turb = np.sqrt(2./3.)*(Rcc/rCGM)**(1./3.)*50.
       b_tot = np.sqrt(b_dop**2. + b_turb**2.)
       
       X,Y,Z =  np.loadtxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), unpack=True)
       
       fl = open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),'rb') 
       data_cl = pickle.load(fl)
       fl.close()  
       '''
       fl = open("./data/cden_pl_nvar_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),'rb') 
       data_cl = pickle.load(fl)
       fl.close() 
       '''
       for key_val in data_cl.keys():
           exec(f"{key_val} = data_cl[\"{key_val}\"]")
       res = res 
       
       with multiprocessing.Pool(processes=proc) as pool:
            result = pool.map(get_EW,res)    
            
       EW = np.array(result)
       
       np.savetxt("./data/ew_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), EW)  
       #np.savetxt("ew_nvar_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), EW)
       #np.savetxt("ew_noturb_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), EW)      
       
       print('complete!')  
    

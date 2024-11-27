# Generate spherical cloud complexes with power-law distribution

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

proc = 10

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

nielsen = lambda r : np.exp(0.27 - 0.015*r)
nielsen_l = lambda r : np.exp(0.27-0.11 + (-0.015-0.002)*r)
nielsen_u = lambda r : np.exp(0.27+0.11 + (-0.015+0.002)*r)

huang = lambda r : np.exp(1.35 - 1.05*np.log10(r) + 0.21*(np.log10(M_star) - 10.3))
huang_l = lambda r : np.exp(1.35-0.25 - (1.05+0.17)*np.log10(r) + (0.21-0.08)*(np.log10(M_star) - 10.3))
huang_u = lambda r : np.exp(1.35+0.25 - (1.05-0.17)*np.log10(r) + (0.21+0.08)*(np.log10(M_star) - 10.3))

dutta = lambda r : np.exp(-0.61 - 0.008*r + 0.53*(np.log10(M_star) - 9.3))
dutta_l = lambda r : np.exp(-0.61-0.58 - (0.008+0.003)*r + (0.53-0.28)*(np.log10(M_star) - 9.3))
dutta_u = lambda r : np.exp(-0.61+0.63 - (0.008-0.004)*r + (0.53+0.34)*(np.log10(M_star) - 9.3))

def power_law(a_l, a_u, a0, alpha, size):    
    cpdf_samples = np.random.uniform(low=0.0, high=1.0, size=size)
    return a0*((a_l/a0)**(3-alpha)*(1-cpdf_samples) + cpdf_samples*(a_u/a0)**(3-alpha))**(1/(3-alpha))

def cloud_gen(Mcold,Ncc,Rcc):
    
    alpha = 1.2
    r0 = 1.
    a_l = 10
    a_u = rCGM-Rcc
    Mcc = Mcold/Ncc
    
    b_turb = np.sqrt(2./3.)*(Rcc/rCGM)**(1./3.)*50.
    b_tot = np.sqrt(b_dop**2. + b_turb**2.)
    
    density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp)
    print('cold gas number density : ',density)
    print('volume fraction (%) : ', Ncc*(Rcc/rCGM)**3.*1.e2)
    print('area covering fraction : ', Ncc*(Rcc/rCGM)**2.)
  
    r = power_law(a_l, a_u, r0, alpha, int(Ncc))
    phi = 2.*np.pi*np.random.uniform(low=0.0, high=1.0, size=int(Ncc))
    theta = np.arccos(2.*np.random.uniform(low=0.0, high=1.0, size=int(Ncc)) -1.)
    X = r*np.sin(theta)*np.cos(phi) 
    Y = r*np.sin(theta)*np.sin(phi) 
    Z = r*np.cos(theta) 

    np.savetxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), np.column_stack((X,Y,Z)))
     
def cloud_intersect(data_cl):
    xl = data_cl[0]
    yl = data_cl[1]
    det = Rcc**2. - (xl-X)**2. - (yl-Y)**2.
    condition = det>=0.
    cl_indx = cloud_index[condition]
    det = det[condition]
    if len(det)==0:
       return 0
    else :
       c_len = 2.*np.sqrt(det)
       c_den = c_len*density*mg_factor*kpc
     
       return cl_indx,c_den

if __name__ == "__main__":

   Mcold_ = [1.e9,1.e10,1.e11]
   Ncc_ = [1.e2,1.e3,1.e4]
   Rcc_ = [5,10,20]
   
   for condition in product(Mcold_,Ncc_,Rcc_):
       
       cloud_gen(*condition)
       
       Mcold = condition[0]
       Ncc = condition[1]
       Rcc = condition[2]
       Mcc = Mcold/Ncc
       
       density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp) 
       
       X,Y,Z =  np.loadtxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), unpack=True)
       
       r = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
       th = np.random.uniform(low=0, high=2.*np.pi, size=int(Nlos))
       xlos = r*np.cos(th)
       ylos = r*np.sin(th)
    
       cloud_index = np.arange(0,int(Ncc),1)
       data_cl = np.column_stack((xlos,ylos))
       with multiprocessing.Pool(processes=proc) as pool:
          res = pool.map(cloud_intersect,data_cl)
    
       data_dump =  {"res": res}
    
       with open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), 'wb') as fl:
          pickle.dump(data_dump, fl)

       
       print('complete! \n')
       

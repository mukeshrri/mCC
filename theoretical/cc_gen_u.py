# Generate spherical cloud complexes with uniform distribution

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

def cloud_gen(Mcold,Ncc,Rcc):

 Mcc = Mcold/Ncc

 density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp)
 print('cold gas number density : ',density)
 print('volume fraction is (%) : ', Ncc*(Rcc/rCGM)**3.*1e2)
 print('area covering fraction is : ', Ncc*(Rcc/rCGM)**2.)

 X = np.zeros(int(Ncc))
 Y = np.zeros(int(Ncc))
 Z = np.zeros(int(Ncc))
 for i in range(int(Ncc)):
    X[i] = np.random.uniform(low=-rCGM, high=rCGM, size=1)[0]
    Y[i] = np.random.uniform(low=-rCGM, high=rCGM, size=1)[0]
    Z[i] = np.random.uniform(low=-rCGM, high=rCGM, size=1)[0]
    dist = np.sqrt(X[i]**2. + Y[i]**2. + Z[i]**2.) 
    while dist>rCGM-Rcc or dist<10.:
        X[i] = np.random.uniform(low=-rCGM, high=rCGM, size=1)[0]
        Y[i] = np.random.uniform(low=-rCGM, high=rCGM, size=1)[0]
        Z[i] = np.random.uniform(low=-rCGM, high=rCGM, size=1)[0]
        dist = np.sqrt(X[i]**2. + Y[i]**2. + Z[i]**2.) 

 print('cloud complex generation complete!')

 np.savetxt("./data/coord_u_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), np.column_stack((X,Y,Z)))

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
       
       X,Y,Z =  np.loadtxt("./data/coord_u_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), unpack=True)
       
       r = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
       th = np.random.uniform(low=0, high=2.*np.pi, size=int(Nlos))
       xlos = r*np.cos(th)
       ylos = r*np.sin(th)
    
       cloud_index = np.arange(0,int(Ncc),1)
       data_cl = np.column_stack((xlos,ylos))
       with multiprocessing.Pool(processes=proc) as pool:
          res = pool.map(cloud_intersect,data_cl)
    
       data_dump =  {"res": res}
    
       with open("./data/cden_u_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), 'wb') as fl:
          pickle.dump(data_dump, fl)

       print('complete! \n')
       

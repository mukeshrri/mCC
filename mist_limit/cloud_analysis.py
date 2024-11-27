# Getting the column density along LOSs
# Next:- go to vpfit_fast.py

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

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

mu = 0.6
mp = 1.6726e-24
Msun = 1.989e33
pc = 3.086e18
kpc = 1.e3*pc
proc = 6

mg_factor = 1.7e-6
N_los = 1.e4
   
def intersect_clouds(data_cl):   
    xl = data_cl[0]
    yl = data_cl[1]
       
    det = rc**2. - ((xl-X)*1.e3)**2. - ((yl-Y)*1.e3)**2.
    condition = det>=0.
    cl_indx = cloud_index[condition]
    det = det[condition]
    if len(det)==0:
       return 0
    else :
      c_len = 2.*np.sqrt(det)
      c_den = c_len*n_cold*mg_factor*pc
      
      return cl_indx,c_den

if __name__ == "__main__":
    n_cold_ = [0.01] # cm^-3
    Rcc_ = [15] # kpc
    rc_ = [60,40,20,15,10] # pc
    Mcc_ = [1e7] # solar mass
       
    for condition in product(n_cold_, 
                             Rcc_,
                             rc_,
                             Mcc_):
                              
        n_cold = condition[0] 
        Rcc = condition[1] 
        rc = condition[2]
        Mcc = condition[3]
        
        with h5py.File("./data/data_{}_{}_{}_{}.hdf5".format(n_cold,Rcc,rc,np.log10(Mcc)), 'r') as f:
             X = np.array(f['X'])
             Y = np.array(f['Y'])
             Z = np.array(f['Z'])
             info = np.array(f['info'])
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = np.array(Z, dtype=np.float32)
        
        Nc = info[4]
        print(Nc)
           
        r_los = np.logspace(-1,np.log10(Rcc),int(N_los))  
        phi = np.random.uniform(low=0, high=2*np.pi, size=int(N_los)) 
        x_los = r_los*np.cos(phi)
        y_los = r_los*np.sin(phi)
        
        cloud_index = np.arange(0,int(Nc),1)

        data_cl = np.column_stack((x_los,y_los))

        with multiprocessing.Pool(processes=proc) as pool:
             res = pool.map(intersect_clouds,data_cl)    
           
        data_dump =  {"res": res}
        
        with open("./data/data_{}_{}_{}_{}.pickle".format(n_cold,Rcc,rc,np.log10(Mcc)), 'wb') as fl:
            pickle.dump(data_dump, fl)
            
        print('complete! \n')
        print("--- %s seconds ---\n " % (time.time() - start_time)) 

        

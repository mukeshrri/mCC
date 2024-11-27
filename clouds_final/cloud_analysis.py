# Get the intersected cloudlets and column density

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
kpc = 3.086e21
fac = mu*mp*kpc**3/Msun
proc = 5

N_los = 1.e4
   
def intersect_clouds(data_cl):
       
          los_x = data_cl[0]
          los_y = data_cl[1]
       
          condition_c = np.logical_and(x0>los_x-a_max, x0<los_x+a_max)
          condition_c = np.logical_and(condition_c, y0>los_y-a_max, y0<los_y+a_max)
          Angl_alfa = angl_alfa[condition_c]
          Angl_beta = angl_beta[condition_c]
          Angl_gmma = angl_gmma[condition_c]
          X0 = x0[condition_c]
          Y0 = y0[condition_c]
          Z0 = z0[condition_c]
          A = a[condition_c]
          B = b[condition_c]
          C = c[condition_c]
          cl_indx = cloud_index[condition_c]
          
          X = np.cos(Angl_beta)*np.cos(Angl_gmma)*(los_x-X0) + np.cos(Angl_beta)*np.sin(Angl_gmma)*(los_y-Y0)
          Y = (np.sin(Angl_alfa)*np.sin(Angl_beta)*np.cos(Angl_gmma)- np.cos(Angl_alfa)*np.sin(Angl_gmma))*(los_x-X0) + (np.sin(Angl_alfa)*np.sin(Angl_beta)*np.sin(Angl_gmma)+ np.cos(Angl_alfa)*np.cos(Angl_gmma))*(los_y-Y0)
          Z = (np.cos(Angl_alfa)*np.sin(Angl_beta)*np.cos(Angl_gmma)+ np.sin(Angl_alfa)*np.sin(Angl_gmma))*(los_x-X0) + (np.cos(Angl_alfa)*np.sin(Angl_beta)*np.sin(Angl_gmma)- np.sin(Angl_alfa)*np.cos(Angl_gmma))*(los_y-Y0)
          
          a_ = (np.sin(Angl_beta)/A)**2. + (np.sin(Angl_alfa)*np.cos(Angl_beta)/B)**2. + (np.cos(Angl_alfa)*np.cos(Angl_beta)/C)**2.
          b_ = 2.*(np.sin(Angl_beta)*X/A**2. + np.sin(Angl_alfa)*np.cos(Angl_beta)*Y/B**2. + np.cos(Angl_alfa)*np.cos(Angl_beta)*Z/C**2.)
          c_ = (X/A)**2. + (Y/B)**2. + (Z/C)**2. - 1. 
          
          det = b_**2. - 4.*a_*c_
          
          condition_int = det > 0  
          cl_indx = cl_indx[condition_int]
          b_ = b_[condition_int]
          a_ = a_[condition_int]

          if len(a_) == 0 :
             root = 0
             cl_indx = 0
             wtr = 0
          else :   
             root = np.sqrt(det[condition_int])/a_
             if np.isinf(root).any() == True:
                print('problem!!!!!!')
                print(det[condition_int])
                print(a_)
                sys.exit()
             wtr = root,cl_indx   
      
          return wtr

if __name__ == "__main__":
    rCGM_ = [280]
    M_cold_ = [1.e10]
    n_cold_ = [0.01]
    N_cc_ = [1000]
    D_min_ = [10]
    D_max_ = [280]
    alpha_ = [0.2]
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
        
        with h5py.File("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.hdf5".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), 'r') as f:
             #X0 = np.array(f['X0'])
             #Y0 = np.array(f['Y0'])
             #Z0 = np.array(f['Z0'])
             x0 = np.array(f['x0'])
             y0 = np.array(f['y0'])
             z0 = np.array(f['z0'])
             a = np.array(f['a'])
             b = np.array(f['b'])
             c = np.array(f['c'])
             angl_alfa = np.array(f['angl_alfa'])
             angl_beta = np.array(f['angl_beta'])
             angl_gmma = np.array(f['angl_gmma'])
             info = np.array(f['info'])
        
        x0 = np.array(x0, dtype=np.float32)
        y0 = np.array(y0, dtype=np.float32)
        z0 = np.array(z0, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)
        angl_alfa = np.array(angl_alfa, dtype=np.float32)
        angl_beta = np.array(angl_beta, dtype=np.float32)
        angl_gmma = np.array(angl_gmma, dtype=np.float32)
        
        N_cl = info[5]
        print(N_cl)
           
        with h5py.File("./data/los_data_4.hdf5", 'r') as f:
             x_los = np.array(f['x_los'])
             y_los = np.array(f['y_los'])
           
        cloud_index = np.arange(0,int(N_cl),1)

        data_cl = np.column_stack((x_los,y_los))

        with multiprocessing.Pool(processes=proc) as pool:
             res = pool.map(intersect_clouds,data_cl)    
           
        data_dump =  {"res": res}
        
        with open("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pickle".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), 'wb') as fl:
            pickle.dump(data_dump, fl)
            
        print('complete! \n')
        print("--- %s seconds ---\n " % (time.time() - start_time)) 
        

# Bottom panel of Fig 10 in the paper

import os
import sys
import numpy as np
from scipy.optimize import root
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
import absorption_calculator as absorption
import velocity_generator as turb_vel
from scipy.optimize import minimize
import scipy.signal as signal

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

mu = 0.6
mp = 1.6726e-24
Msun = 1.989e33
pc = 3.086e18
kpc = 1.e3*pc
kms = 1.e5
kB = 1.3807e-16
amu = 1.6737e-24
c = 2.9979e10

Temp = 1.e4
M_star = 6.08e10
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
a_ion = 24.305 
mass_ion = a_ion*amu
b_dopp = np.sqrt(2*kB*Temp/mass_ion)/kms # doppler width in km/s
wave0 = 2796.3553 # rest-frame transition in Angstrom
spectral_res = 0.001 # spectral resolution in km/s
    
N_los = 1.e4

n_cold = 0.01
Rcc = 10
rc = 20
Mcc = 1e7
rCGM = 280.
vmax = np.sqrt(2.)*50.*(Rcc/rCGM)**(1./3.)

pertx,perty,pertz,k_mode = turb_vel.velocity_generator(n=[200,200,200], kmin=1, kmax=100, beta=1./3., vmax=vmax)

X=Y=Z = np.linspace(-Rcc, Rcc, 200)
  
with h5py.File("./data/data_{}_{}_{}_{}.hdf5".format(n_cold,Rcc,rc,np.log10(Mcc)),'r') as f:
     X0 = np.array(f['X'])
     Y0 = np.array(f['Y'])
     Z0 = np.array(f['Z'])
     info = np.array(f['info'])

Nc = info[4] 
print('no of cloudlets:', Nc) 

cloud_index = np.arange(0,int(Nc),1)
color = ['grey','maroon','red','yellow','lime','green','cyan','black','navy','blue',
'indigo','magenta','lightpink','teal','plum','brown','khaki','tan',
'aliceblue','pink','purple','tomato','silver','rosybrown']

fl = open("./data/data_{}_{}_{}_{}.pickle".format(n_cold,Rcc,rc,np.log10(Mcc)),'rb')
data_cl = pickle.load(fl)
fl.close()  

for key_val in data_cl.keys():
    exec(f"{key_val} = data_cl[\"{key_val}\"]")
res = res 

r_los = np.logspace(-1,np.log10(Rcc),int(N_los)) 

counter = 0

for i in range(1,int(N_los),1):
   if_tuple = isinstance(res[i], tuple)
   if if_tuple:
      counter += 1
     
      ress = res[i]
      cnt = np.array(ress[0]).shape[0]
      print('count : ', cnt)
      arr = np.array(ress[1])
      arr1 = np.array(ress[0])
      if np.isinf(arr).any() == True:
          print('problem at : ')
          print(arr)
          print(np.array(ress[1]))
          sys.exit()
      col_den = arr
      
      ################### calculate b_thermal for individual cloudlets intersected
      v_los = np.zeros(int(cnt))
      z_los = np.zeros(int(cnt))
      fig, axs = plt.subplots(figsize=(10,8))
      
      for j in range(int(cnt)):
         x_indx = np.argmin(np.abs(X - X0[int(arr1[j])]))
         y_indx = np.argmin(np.abs(Y - Y0[int(arr1[j])]))
         z_indx = np.argmin(np.abs(Z - Z0[int(arr1[j])]))
         v_los[j] = pertz[x_indx,y_indx,z_indx]
         z_los[j] = Z0[j]
      
         axs.scatter(v_los[j], z_los[j], s=600, edgecolors='black', color=color[j])
      
      axs.grid()
      axs.tick_params('both', length=8, width=3, which='major', labelsize=20)
      axs.tick_params('both', length=4, width=2, which='minor', labelsize=20) 
      axs.set_xlabel(r'$v_{\rm los} \, \rm (km \, s^{-1}$)', fontsize=24)
      axs.set_ylabel(r'$z$ (kpc)', fontsize=24)
      axs.set_xlim(-25,35)
      axs.set_ylim(-Rcc,Rcc) 
      fig.tight_layout()  
      plt.savefig('./figures/zvz.pdf')
      plt.show()
      plt.close()  
      
print('no of empty sightlines', N_los-counter)         
print('complete! \n')
print("--- %s seconds ---\n " % (time.time() - start_time)) 


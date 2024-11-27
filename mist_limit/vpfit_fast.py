# Fit the required number of Voight profile and get sigma turb
# Next:- go to sigmaturb.py

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

proc = 6

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
Rcc = 15
rc = 10
Mcc = 1e7
rCGM = 280
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
color = ['green','deepskyblue','magenta','maroon','olive','blue','peru','orange',
'lightcoral','lawngreen','plum','tan','aliceblue','gray','coral','khaki','tomato',
'wheat','gold','lightpink','lightcyan','linen']

def turb_cal(ress):
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
      
      ################### calculate b_thermal for individual cloudlets intersected
      v_los = np.zeros(int(cnt))
      
      for j in range(int(cnt)):
         x_indx = np.argmin(np.abs(X - X0[int(arr1[j])]))
         y_indx = np.argmin(np.abs(Y - Y0[int(arr1[j])]))
         z_indx = np.argmin(np.abs(Z - Z0[int(arr1[j])]))
         v_los[j] = pertz[x_indx,y_indx,z_indx]
      
      vel_all = np.linspace(min(v_los)-3.*b_dopp, max(v_los)+3.*b_dopp, 100)
      tau_total = np.zeros(vel_all.shape[0])
      
      for j in range(int(cnt)):
         N_l = col_den[j]
         nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_dopp, wave0, v_los[j], spectral_res)
         norm_flux = absorption.generate_norm_flux(op_depth)
         wavel = c/nu * 1.e8
         vel_range = (wavel-wave0)/wave0 * c /kms
         
         tau_total += np.interp(vel_all, vel_range, op_depth)
          
      flux_total = np.exp(-tau_total) 
      
      data = flux_total
      fluxd = data
      veld = vel_all
      slope = (fluxd[1:]-fluxd[:-1])/(veld[1:]-veld[:-1])
      vels = (veld[1:]+veld[:-1])/2.
      '''
      plt.plot(vels, slope, 'r-', markersize=2)
      plt.grid()
      plt.plot(veld,fluxd, 'b-', markersize=2)
      plt.show()
      '''
      locmin_arg = signal.argrelmin(fluxd)
      vlos = veld[locmin_arg]

      locmin_slp = signal.argrelmin(slope, order=1)
      locmin_slp_arg = np.array(locmin_slp[0])
      condition = slope[locmin_slp_arg]>0
      locmin_slp_arg = locmin_slp_arg[condition]
      vlos_mn = vels[locmin_slp_arg]

      locmax_slp = signal.argrelmax(slope, order=1)
      locmax_slp_arg = np.array(locmax_slp[0])
      condition = slope[locmax_slp_arg]<0
      locmax_slp_arg = locmax_slp_arg[condition]
      vlos_mx = vels[locmax_slp_arg]

      no_vf = len(locmin_arg[0]) + locmin_slp_arg.shape[0] + locmax_slp_arg.shape[0]
      print(no_vf)

      v_los = []
      for j in range(len(locmin_arg[0])):
          v_los.append(vlos[j])
      for j in range(locmin_slp_arg.shape[0]):
          v_los.append(vlos_mn[j])
      for j in range(locmax_slp_arg.shape[0]):
          v_los.append(vlos_mx[j])

      v_los = np.sort(v_los)

      def fun(x):
    
          vel_all = np.linspace(min(veld), max(veld), 200)
          tau_total = np.zeros(vel_all.shape[0])
    
          for j in range(no_vf):
            bt = np.sqrt(x[3*j+1]**2. + b_dopp**2.)
            nu, op_depth = absorption.return_optical_depth(f_lu, 10**x[3*j], gamma_ul, bt, wave0, x[3*j+2], spectral_res)
            wavel = c/nu * 1.e8
            vel_range = (wavel-wave0)/wave0 * c /kms
            tau_total += np.interp(vel_all, vel_range, op_depth)
          flux_total = np.exp(-tau_total) 
          model = np.interp(veld, vel_all, flux_total)
    
          wtr = np.sum((data-model)**2.)
    
          return wtr

      nll = lambda *args: fun(*args)
      ini_guess = []
      for j in range(no_vf):
         ini_guess.append(12.)
         ini_guess.append(1.)
         ini_guess.append(v_los[j])
      ini_guess = np.array(ini_guess)

      bound = ((7.,20.),(0,20.), (-100.,100.)) * no_vf

      sol = minimize(nll, ini_guess, tol=1.e-6, bounds=bound, options={"disp":False})

      param_opt = sol.x
      return param_opt
   
   else:
       return 0   
######################################################

fl = open("./data/data_{}_{}_{}_{}.pickle".format(n_cold,Rcc,rc,np.log10(Mcc)),'rb')
data_cl = pickle.load(fl)
fl.close()  

for key_val in data_cl.keys():
    exec(f"{key_val} = data_cl[\"{key_val}\"]")
res = res 

with multiprocessing.Pool(processes=proc) as pool:
     result = pool.map(turb_cal,res)   

with open("./data/dataturb_{}_{}_{}_{}.pickle".format(n_cold,Rcc,rc,np.log10(Mcc)), 'wb') as fl:
     pickle.dump(result, fl)
            
print('complete! \n')
print("--- %s seconds ---\n " % (time.time() - start_time)) 


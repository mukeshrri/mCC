# Top panel of Fig 10 in paper 

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
nu0 = c / (wave0*1.e-8)
spectral_res = 0.001 # spectral resolution in km/s
spect_res = 6
    
N_los = 1.e4

n_cold = 0.01
Rcc = 10
rc = 20
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
    
fl = open("./data/data_{}_{}_{}_{}.pickle".format(n_cold,Rcc,rc,np.log10(Mcc)),'rb')
data_cl = pickle.load(fl)
fl.close()  

for key_val in data_cl.keys():
    exec(f"{key_val} = data_cl[\"{key_val}\"]")
res = res 
     
r_los = np.logspace(-1,np.log10(Rcc),int(N_los))  
phi = np.random.uniform(low=0, high=2*np.pi, size=int(N_los)) 
x_los = r_los*np.cos(phi)
y_los = r_los*np.sin(phi)
 
cloud_index = np.arange(0,int(Nc),1)
color = ['grey','maroon','red','yellow','lime','green','cyan','black','navy','blue',
'indigo','magenta','lightpink','teal','plum','brown','khaki','tan','thistle',
'aliceblue','pink','purple','tomato','silver','rosybrown','fuchsia','indigo','orchid']

for i in range(1,r_los.shape[0],1):
   print("i = ", i)
   if_tuple = isinstance(res[i], tuple)
   if if_tuple:
      cnt = np.array(res[i][0]).shape[0]
      arr = np.array(res[i][1])
      arr1 = np.array(res[i][0])
      if np.isinf(arr).any() == True:
          print('problem at : ', i)
          print(arr)
          print(np.array(res[i][1]))
          sys.exit()
      col_den = arr
      print(cnt)
      ################### calculate b_thermal for individual cloudlets intersected
      v_los = np.zeros(int(cnt))
      
      for j in range(int(cnt)):
         x_indx = np.argmin(np.abs(X - X0[int(arr1[j])]))
         y_indx = np.argmin(np.abs(Y - Y0[int(arr1[j])]))
         z_indx = np.argmin(np.abs(Z - Z0[int(arr1[j])]))
         v_los[j] = pertz[x_indx,y_indx,z_indx]
      
      vel_all = np.linspace(min(v_los)-3.*b_dopp, max(v_los)+3.*b_dopp, 100)
     
      tau_total = np.zeros(vel_all.shape[0])
      
      fig, axs = plt.subplots(figsize=(10,8))
      
      for j in range(int(cnt)):
         N_l = col_den[j]
         nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_dopp, wave0, v_los[j], spectral_res)
         norm_flux = absorption.generate_norm_flux(op_depth)
         wavel = c/nu * 1.e8
         vel_range = (wavel-wave0)/wave0 * c /kms
         
         axs.plot(vel_range, np.exp(-op_depth), color=color[j], ls='--', lw=6, zorder=100)
         
         tau_total += np.interp(vel_all, vel_range, op_depth)
          
      flux_total = np.exp(-tau_total) 
      
      nu = nu0 * (1. - vel_all*kms/c)
      
      axs.plot(vel_all, flux_total, color='orange', ls='-', lw=6, label='overall profile')
      
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

      bound = ((8.,18.),(0,10.), (-100.,100.)) * no_vf

      sol = minimize(nll, ini_guess, tol=1.e-6, bounds=bound, options={"disp":True})

      param_opt = sol.x
      #print(param_opt)
      
      vel_all1 = np.linspace(min(veld), max(veld), 100)
      tau_total1 = np.zeros(vel_all1.shape[0])
      '''
      for j in range(no_vf):
          N_l = 10**param_opt[3*j]
          btot = np.sqrt(param_opt[3*j+1]**2. + b_dopp**2.)
          nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, btot, wave0, param_opt[3*j+2], spectral_res)
          wavel = c/nu * 1.e8
          vel_range = (wavel-wave0)/wave0 * c /kms
          
          if  j ==0 :
              axs.plot(vel_range, np.exp(-op_depth), color='deepskyblue', ls='--', lw=4, label='fitted components')
          else :
              axs.plot(vel_range, np.exp(-op_depth), color='deepskyblue', ls='--', lw=4)
              
          tau_total1 += np.interp(vel_all1, vel_range, op_depth)
      
      flux_total1 = np.exp(-tau_total1) 
      
      axs.plot(vel_all1, flux_total1, color='deepskyblue', ls=':', lw=6, label='overall fitted profile')
      '''
      #plt.title(r"Nlos={},Nfit={},$R_{{\perp}}$={:.4f} kpc".format(cnt,no_vf,r_los[i], color='black', fontsize=20))
      axs.set_xlim(min(veld), max(veld))
      axs.set_ylim(0.,1.1)
      axs.grid()
      axs.tick_params('both', length=8, width=3, which='major', labelsize=20)
      axs.tick_params('both', length=4, width=2, which='minor', labelsize=20) 
      axs.set_xlabel(r'$V_{\rm los} \, (\rm km \, s^{-1})$', fontsize=24)
      axs.set_ylabel(r'Normalized absorption profile', fontsize=24)
      axs.legend(loc='upper center', ncol=2, fontsize=20)
      fig.tight_layout()
      plt.savefig('./figures/sturb_best.pdf')
      plt.show()
      plt.close()   
      

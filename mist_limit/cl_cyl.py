# cloudlets generated in a cylindrical volume (Fig 12 in paper)
# Next:- go to cloud_analysis.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from decimal import Decimal
import multiprocessing
import warnings
import time
from itertools import product
import pickle
from tqdm import tqdm
import random
import absorption_calculator as absorption
import velocity_generator as turb_vel
import scipy.signal as signal
from scipy.optimize import minimize
from scipy.optimize import root

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
rCGM = 280

Temp = 1.e4
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
a_ion = 24.305 
mass_ion = a_ion*amu
b_dopp = np.sqrt(2*kB*Temp/mass_ion)/kms # doppler width in km/s
wave0 = 2796.3553 # rest-frame transition in Angstrom
nu0 = c / (wave0*1.e-8)
spectral_res = 0.001 # spectral resolution in km/s
spect_res = 6
v_max = 50.

mg_factor = 1.7e-6
distri = 'u'

color = ['grey','maroon','red']

def cloud_gen(n_cold, 
              Rcc,
              rc,
              Mcc,
              Rcyl,
              Hcyl):   

   print('n_cold:', n_cold,
         'Rcc:', Rcc,
         'rc:', rc,
         'Mcc:', Mcc,
         'Rcyl:', Rcyl,
         'Hcyl:', Hcyl)
   
   Mcyl = (np.pi*Rcyl**2*Hcyl)/(4./3.*np.pi*Rcc**3.) * Mcc
   Nc = Mcyl*Msun/(4./3.*np.pi*(rc*pc)**3.*n_cold*mu*mp)
   print('No of cloudlets are : ', Nc)
         
   # generate cloudlets
   X = np.zeros(int(Nc))
   Y = np.zeros(int(Nc))
   Z = np.zeros(int(Nc))
   
   xl = np.sqrt(Rcc**2. - (Hcyl/2.)**2.)
   yl = 0.
   
   if distri == 'u':
     for i in range(int(Nc)):
      X[i] = np.random.uniform(low=-Rcyl, high=Rcyl, size=1)[0]
      Y[i] = np.random.uniform(low=-Rcyl, high=Rcyl, size=1)[0]
      Z[i] = np.random.uniform(low=-Hcyl/2., high=Hcyl/2., size=1)[0]
      dist = np.sqrt(X[i]**2. + Y[i]**2.) 
      while dist>Rcyl:
         X[i] = np.random.uniform(low=-Rcyl, high=Rcyl, size=1)[0]
         Y[i] = np.random.uniform(low=-Rcyl, high=Rcyl, size=1)[0]
         dist = np.sqrt(X[i]**2. + Y[i]**2.)
   
   if distri == 'pl':
   
     r = power_law(D_min, D_max, 10.0, alpha, int(Nc))
     phi = 2.*np.pi*np.random.uniform(low=0.0, high=1.0, size=int(Nc))
     theta = np.arccos(2*np.random.uniform(low=0.0, high=1.0, size=int(Nc)) -1.)
     X = r*np.sin(theta)*np.cos(phi) 
     Y = r*np.sin(theta)*np.sin(phi) 
     Z = r*np.cos(theta) 
   
   X = X+xl
   Y = Y+yl
   
   print('complete! \n')
   
   cloud_index = np.arange(0,int(Nc),1)
   
   det = rc**2. - ((xl-X)*1.e3)**2. - ((yl-Y)*1.e3)**2.
   condition = det>=0.
   z_pos = Z[condition]
   arr1 = cloud_index[condition]
   det = det[condition]
   c_len = 2.*np.sqrt(det)
   c_den = c_len*n_cold*mg_factor*pc
   
   print('total :', np.log10(np.sum(c_den)))
   
   vmax = np.sqrt(2.)*50.*(Rcc/rCGM)**(1./3.)
   
   pertx,perty,pertz,k_mode = turb_vel.velocity_generator(n=[200,200,200], kmin=1, kmax=100, beta=1./3., vmax=vmax)
   
   X0=Y0=Z0 = np.linspace(-Rcc, Rcc, 200)
   
   cnt = c_den.shape[0]
   print('no of cl intersected are :', cnt)
   v_los = np.zeros(int(cnt))
   
   for j in range(int(cnt)):
         x_indx = np.argmin(np.abs(X0 - X[int(arr1[j])]))
         y_indx = np.argmin(np.abs(Y0 - Y[int(arr1[j])]))
         z_indx = np.argmin(np.abs(Z0 - Z[int(arr1[j])]))
         v_los[j] = pertz[x_indx,y_indx,z_indx]
   
   vel_all = np.linspace(min(v_los)-4.*b_dopp, max(v_los)+4.*b_dopp, 100)
   
   tau_total = np.zeros(vel_all.shape[0])
   
   fig, axs = plt.subplots(figsize=(10,8))
   ew = 0
      
   for j in range(int(cnt)):
         N_l = c_den[j]
         nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_dopp, wave0, v_los[j], spectral_res)
         norm_flux = absorption.generate_norm_flux(op_depth)
         wavel = c/nu * 1.e8

         EW = absorption.return_EW(wavel, norm_flux)
         ew += EW
         
         vel_range = (wavel-wave0)/wave0 * c /kms
         
         axs.plot(vel_range, np.exp(-op_depth), color=color[0], ls='--', lw=2)
         
         tau_total += np.interp(vel_all, vel_range, op_depth)
          
   flux_total = np.exp(-tau_total) 
      
   nu = nu0 * (1. - vel_all*kms/c)
      
   wave_coarse, flux_coarse = absorption.return_coarse_flux(nu,flux_total,spect_res,wave0)
      
   vel_coarse = (wave_coarse/wave0 - 1.)*c/kms
   
   axs.plot(vel_all, flux_total, color='orange', ls='-', lw=6, label='overall profile')
      
   axs.step(vel_coarse, flux_coarse, where='mid', color='blue', ls='-', lw=6, label='coarse profile')
   
   #####################
   N_l = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3. * mu*mp) * Hcyl*kpc * mg_factor
   print('Misty CC estimate', np.log10(N_l))
   b_turb = np.sqrt(2./3.)*(Hcyl/rCGM)**(1./3.) * v_max
   b_tot = np.sqrt(b_dopp**2. + b_turb**2.)
   vlos = (max(v_los)+min(v_los))/2.
   nu, op_depth = absorption.return_optical_depth(f_lu, N_l, gamma_ul, b_tot, wave0, vlos, spectral_res)
   norm_flux = absorption.generate_norm_flux(op_depth)
   wavel = c/nu * 1.e8
   vel_range = (wavel-wave0)/wave0 * c /kms
   
   axs.plot(vel_range, norm_flux, color='magenta', ls='--', lw=6, label='CC')
   #####################   
      
   axs.set_title(r"$N_{{\rm cl,los}}$={}, $r_{{\rm cl}}$={} pc".format(cnt,rc, color='black', fontsize=28))
   axs.title.set_size(28)
   axs.set_xlim(min(v_los)-15,max(v_los)+15)
   axs.set_ylim(0.,1.01)
   axs.grid()
   axs.tick_params('both', length=8, width=3, which='major', labelsize=22)
   axs.tick_params('both', length=4, width=2, which='minor', labelsize=22) 
   axs.set_xlabel(r'$V_{\rm los} \, (\rm km \, s^{-1})$', fontsize=25)
   axs.set_ylabel(r'Normalized absorption profile', fontsize=25)
   fig.tight_layout()
   #plt.savefig('./figures/ab_profile_cyl_{}_{}.png'.format(rc,Hcyl))
   plt.savefig('./figures/ab_profile_cyl_{}_{}.pdf'.format(rc,Hcyl))
   plt.show()
   plt.close()    
   
   fig, axs = plt.subplots(figsize=(10,8))
   
   axs.scatter(v_los, z_pos, color='grey', s=200)
   
   axs.set_xlabel(r'$V_{\rm los} \, (\rm km \, s^{-1})$', fontsize=25)
   axs.set_ylabel(r'Z (kpc)', fontsize=25)
   #axs.legend(loc='upper center', ncol=2, fontsize=19)
   axs.tick_params('both', length=8, width=3, which='major', labelsize=22)
   axs.tick_params('both', length=4, width=2, which='minor', labelsize=22)
   axs.grid()
   fig.tight_layout()
   #plt.savefig('./figures/zvz_cyl_{}_{}.png'.format(rc,Hcyl))
   plt.show()
   plt.close()
   
   ##############################
   data = flux_total
   fluxd = data
   veld = vel_all
   slope = (fluxd[1:]-fluxd[:-1])/(veld[1:]-veld[:-1])
   vels = (veld[1:]+veld[:-1])/2.
   
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
   
   sig_turb = []
   col_den = np.zeros((no_vf))
   
   for k in range(int(no_vf)):
       col_den[k] = param_opt[3*k]
       sig_turb.append(param_opt[3*k+1])
   
   print('fitted total :', np.log10(np.sum(col_den)))
   
   sig_turb = np.array(sig_turb)
   
   print('mean = ', np.average(sig_turb))
   print('std = ', np.std(sig_turb))    
   
   fig, axs = plt.subplots(figsize=(10,8))
   
   plt.hist(sig_turb, bins=np.arange(0,max(sig_turb),0.2), align='mid', color='olive', density=False, weights=0.2*np.ones_like(sig_turb)/no_vf, stacked=True, histtype='step', lw=4, ls='-')
   
   axs.set_title('mean={:.1f}, sigma={:.1f}'.format(np.average(sig_turb),np.std(sig_turb)), fontsize=20)
   axs.set_ylim(1.e-3,1.1) 
   axs.set_yscale('log')
   axs.grid()
   axs.tick_params('both', length=6, width=2, which='major', labelsize=15)
   axs.tick_params('both', length=4, width=1, which='minor', labelsize=15) 
   axs.set_xlabel(r'$\sigma_{\rm turb} \, \rm (km \, s^{-1}$)', fontsize=15)
   axs.set_ylabel(r'Normalized frequency', fontsize=15)
   fig.tight_layout()
   #plt.savefig('./figures/sigma_turb_hist_{}.png'.format(rc))
   plt.show()
   plt.close()     
   
if __name__ == "__main__":

    n_cold_ = [0.01] # cm^-3
    Rcc_ = [10] # kpc
    rc_ = [10,1,0.1] # pc
    Mcc_ = [1e7] # solar mass
    Rcyl_ = [0.05] # kpc
    Hcyl_ = [20] # kpc

    for condition in product(n_cold_, 
                             Rcc_,
                             rc_,
                             Mcc_,
                             Rcyl_,
                             Hcyl_):
                             
        cloud_gen(*condition)
            
        print('complete! \n')
        print("--- %s seconds ---\n " % (time.time() - start_time)) 


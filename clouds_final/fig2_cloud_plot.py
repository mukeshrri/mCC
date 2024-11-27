# Bottom panel of Fig 14

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
import absorption_calculator as absorption
import velocity_generator as turb_vel

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
    
N_los = 10000

N_cc = 1000#np.array([100,1000,10000])

M_cold = 1.e10#np.array([1.e9,5.e9,1.e10])
n_cold = 0.01#np.array([0.01, 0.001])
rCGM = 280

D_min = 10
D_max = 280
alpha = 1.2#np.array([0.2,1.2,2.2])

d_min = 0.1
d_max = 10#np.array([5,10,20])
beta = 0.2#np.array([0.2,1.2,2.2])

a_min = 0.01#np.array([0.1,0.01])
a_max = 1
gamma = np.array([0.2,1.2,2.2])

variable = gamma  ################ change here
var_name = 'gamma'  ################ change here

nielsen = lambda r : np.exp(0.27 - 0.015*r)
nielsen_l = lambda r : np.exp(0.27-0.11 + (-0.015-0.002)*r)
nielsen_u = lambda r : np.exp(0.27+0.11 + (-0.015+0.002)*r)

huang = lambda r : np.exp(1.35 - 1.05*np.log10(r) + 0.21*(np.log10(M_star) - 10.3))
huang_l = lambda r : np.exp(1.35-0.25 - (1.05+0.17)*np.log10(r) + (0.21-0.08)*(np.log10(M_star) - 10.3))
huang_u = lambda r : np.exp(1.35+0.25 - (1.05-0.17)*np.log10(r) + (0.21+0.08)*(np.log10(M_star) - 10.3))

dutta = lambda r : np.exp(-0.61 - 0.008*r + 0.53*(np.log10(M_star) - 9.3))
dutta_l = lambda r : np.exp(-0.61-0.58 - (0.008+0.003)*r + (0.53-0.28)*(np.log10(M_star) - 9.3))
dutta_u = lambda r : np.exp(-0.61+0.63 - (0.008-0.004)*r + (0.53+0.34)*(np.log10(M_star) - 9.3))

r_perp = np.linspace(1,rCGM,100)

fig1, axs1 = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
color = np.array(['deepskyblue', 'coral', 'gray'])

for var in range(variable.shape[0]):
  
  gamma = variable[var] ################### change here
  
  MCLD = int(np.log10(M_cold)) if M_cold != 5.e9 else 5.9 
  
  if var_name == 'N_cc':
     label = r'$N_{{\rm cc}}$ = {}'.format(variable[var])
  if var_name == 'gamma':
     label = r'$\gamma $ = {}'.format(variable[var])
  if var_name == 'a_max':  
     label = r'$a_{{\rm max}}$ = {} kpc'.format(variable[var]) 
  if var_name == 'a_min':  
     label = r'$a_{{\rm min}}$ = {:.0f} pc'.format(variable[var]*1.e3) 
  if var_name == 'd_max':  
     label = r'$d_{{\rm max}}$ = {} kpc'.format(variable[var])   
  if var_name == 'M_cold':  
     if M_cold == 5.e9:
        label = r'$\rm log_{10}\ (M_{\rm cold})$ = 9.7'
     else :   
        label = r'$\rm log_{{10}}\ (M_{{\rm cold}}) = {:.0f}$'.format(np.log10(variable[var]))
  if var_name == 'n_cold':  
     label = r'$n_{{\rm cold}} = {}\ \rm cm^{{-3}}$'.format(variable[var]) 
  if var_name == 'alpha':  
     label = r'$\alpha$ = {}'.format(variable[var])     
  if var_name == 'beta':  
     label = r'$\beta$ = {}'.format(variable[var])             
  
  with h5py.File("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.hdf5".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), 'r') as f:
     #X0 = np.array(f['X0'])
     #Y0 = np.array(f['Y0'])
     #Z0 = np.array(f['Z0'])
     x0 = np.array(f['x0'])
     y0 = np.array(f['y0'])
     z0 = np.array(f['z0'])
     #A = np.array(f['a'])
     #B = np.array(f['b'])
     #C = np.array(f['c'])
     #angl_alfa = np.array(f['angl_alfa'])
     #angl_beta = np.array(f['angl_beta'])
     #angl_gmma = np.array(f['angl_gmma'])
     info = np.array(f['info'])
  
  N_cl = info[5]
  print('no of clouds:', N_cl) 
 
  with h5py.File("./data/los_data_4.hdf5", 'r') as f:
     x_los = np.array(f['x_los'])
     y_los = np.array(f['y_los'])
        
  r_los = np.sqrt(x_los**2. + y_los**2.)
  no_sightlines = N_los
  
  Mg_density = 1.7e-6*n_cold
  
  cnt,col_den,EWidth = np.loadtxt("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), unpack=True)
  
  EW_avg = []
  rad_bin = np.arange(0,rCGM+1,20)
  for i in range(rad_bin.shape[0]-1):
     condition = np.logical_and(r_los>rad_bin[i], r_los<=rad_bin[i+1])
     #condition = np.logical_and(condition, n_int!=0)
     EW_avg.append(np.average(EWidth[condition])) 
       
  axs1[var].scatter(r_los, EWidth, s=50, color=color[var], facecolors='None', alpha=0.4) 
  
  axs1[var].plot(r_perp, nielsen(r_perp), lw=3, color='blue', zorder=100, label='Nielsen')
  axs1[var].plot(r_perp, nielsen_l(r_perp), lw=3, color='blue', ls=':', zorder=100) 
  axs1[var].plot(r_perp, nielsen_u(r_perp), lw=3, color='blue', ls=':', zorder=100) 
  
  axs1[var].plot(r_perp, huang(r_perp), lw=3, color='green', zorder=100, label='Huang')  
  axs1[var].plot(r_perp, huang_l(r_perp), lw=3, color='green', ls=':', zorder=100)
  axs1[var].plot(r_perp, huang_u(r_perp), lw=3, color='green', ls=':', zorder=100)
   
  axs1[var].plot(r_perp, dutta(r_perp), lw=3, color='magenta', zorder=100, label='Dutta') 
  axs1[var].plot(r_perp, dutta_l(r_perp), lw=3, color='magenta', ls=':', zorder=100)
  axs1[var].plot(r_perp, dutta_u(r_perp), lw=3, color='magenta', ls=':',zorder=100)
  axs1[var].plot((rad_bin[:-1]+rad_bin[1:])/2., EW_avg, 'k-', linewidth=3, label='mean', zorder=100)
  
  if var==2:
     r_bin,EW_pl = np.loadtxt('mean_EW_pl.txt', unpack=True) # generate this in 'theoretical' folder
      
     axs1[var].plot(r_bin, EW_pl, 'k--', linewidth=3, zorder=100)
       
  axs1[var].set_xlim(10,300) 
  axs1[var].set_ylim(1.e-2,4)
  axs1[var].set_xscale('log')
  axs1[var].set_yscale('log')
  axs1[var].grid(axis='both', which='major')
  axs1[var].set_xlabel(r'$R_\perp$ (kpc)', fontsize=16)
  
  if var !=0 :
        axs1[var].set_yticklabels(labels=[], minor=False)
  
axs1[0].set_ylabel(r'MgII Equivalent Width ($\rm \AA$)', fontsize=15)

axs1[0].tick_params(axis='both', which='major', length=8, width=3, labelsize=13)
axs1[0].tick_params(axis='both', which='minor', length=4, width=2, labelsize=13)
axs1[1].tick_params(axis='x', which='major', length=8, width=3, labelsize=13)
axs1[1].tick_params(axis='x', which='minor', length=4, width=2, labelsize=13)
axs1[1].tick_params(axis='y', which='major', direction='inout', length=8, width=3, labelsize=13)
axs1[1].tick_params(axis='y', which='minor', direction='inout', length=4, width=2, labelsize=13)
axs1[2].tick_params(axis='x', which='major', length=8, width=3, labelsize=13)
axs1[2].tick_params(axis='x', which='minor', length=4, width=2, labelsize=13)  
axs1[2].tick_params(axis='y', which='major', direction='inout', length=8, width=3, labelsize=13)
axs1[2].tick_params(axis='y', which='minor', direction='inout', length=4, width=2, labelsize=13)

axs1[0].legend(fontsize=15)

fig1.tight_layout()
plt.subplots_adjust(wspace=0.0)

fig1.savefig("./figures/figure2_{}.pdf".format(var_name))
plt.show()  
plt.close()

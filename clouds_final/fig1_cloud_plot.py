# Fig 15 in paper

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
n_cold = 0.01#np.array([0.1,0.01,0.005])
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

variable = gamma  ################ change here (with the same name as decalred above to see variation with the parameter)
var_name = 'gamma'  ################ change here

r_perp = np.linspace(1,rCGM,100)

fig1, axs1 = plt.subplots(nrows=1, ncols=4, figsize=(20,5))
color = np.array(['deepskyblue', 'coral', 'gray'])

for var in range(variable.shape[0]):
  
  gamma = variable[var] ################### change here
  
  MCLD = int(np.log10(M_cold)) if M_cold != 5.e9 else 5.9
  
  if var_name == 'N_cc':
     label = r'$N_{{\rm cc}} = 10^{{{}}}$'.format(int(np.log10(variable[var])))
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
     else:
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
  
  with h5py.File("los_data_4.hdf5", 'r') as f:
     x_los = np.array(f['x_los'])
     y_los = np.array(f['y_los'])
        
  r_los = np.sqrt(x_los**2. + y_los**2.)
  no_sightlines = N_los
  
  Mg_density = 1.7e-6*n_cold
  
  cnt,col_den,EWidth = np.loadtxt("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(N_cc,MCLD,n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), unpack=True)
  
  col_den = np.where(col_den==0.0, 0.0, np.log10(col_den))
  
  impact = np.arange(0,220,20)
  cov_frac_b = np.zeros(impact.shape[0]-1)
  impact_b = np.zeros(impact.shape[0]-1)
  for i in range(impact.shape[0]-1):
      condition1 = np.logical_and(r_los > impact[i], r_los <= impact[i+1])
      condition = np.logical_and(condition1, EWidth > 0.3)
      cov_frac_b[i] = EWidth[condition].shape[0]/r_los[condition1].shape[0]
      impact_b[i] = (impact[i] + impact[i+1])/2.
       
  axs1[3].plot(impact_b, cov_frac_b, lw=6, ls='-', color=color[var])        
  
  axs1[0].hist(cnt, bins=np.arange(0,max(cnt),3), density=False, weights=np.ones_like(cnt)/cnt.shape[0], align='mid', histtype='step', color=color[var], stacked=False, log=True, linewidth=6, label=label)   
  
  axs1[1].hist(col_den, bins=np.arange(min(col_den),max(col_den)+1,0.5), density=False, weights=np.ones_like(col_den)/col_den.shape[0], align='mid', histtype='step', color=color[var], stacked=False, log=True, linewidth=6)   
  
  axs1[2].hist(EWidth, bins=np.arange(0,max(EWidth),.05), density=False, weights=np.ones_like(EWidth)/EWidth.shape[0], align='mid', histtype='step', color=color[var], stacked=True, log=True, linewidth=6)  
  
mcc_impact,cov_frac_mcc = np.loadtxt('cov_frac_mCC.txt', unpack=True) # generate this file in the 'theoretical' folder

axs1[3].plot(mcc_impact,cov_frac_mcc, ls='-', lw=6, color='black', label='mCC') 

r_los = np.logspace(np.log10(10),np.log10(rCGM),int(N_los))
EW = np.loadtxt("ew_nvar_10_3_10.txt")
impact = np.arange(0,220,20) 
cov_frac_b = np.zeros(impact.shape[0]-1)
impact_b = np.zeros(impact.shape[0]-1)
for i in range(impact.shape[0]-1):
    condition1 = np.logical_and(r_los > impact[i], r_los <= impact[i+1])
    condition = np.logical_and(condition1, EW > 0.3)
    cov_frac_b[i] = EW[condition].shape[0]/r_los[condition1].shape[0]
    impact_b[i] = (impact[i] + impact[i+1])/2. 
axs1[3].plot(impact_b,cov_frac_b, ls='--', lw=6, color='black')   
 
axs1[3].errorbar(20,0.93, yerr=0.04, xerr=None, ecolor='green', elinewidth=4, zorder=100)
axs1[3].scatter(20,0.93, s=150, color='green', label='Huang', zorder=100) 
axs1[3].errorbar(70,0.36, yerr=0.07, xerr=None, ecolor='green', elinewidth=4, zorder=100)
axs1[3].scatter(70,0.36, s=150, color='green', zorder=100)

axs1[3].errorbar(12.5,0.96, yerr=0.08, xerr=None, ecolor='blue', elinewidth=4, zorder=100)
axs1[3].scatter(12.5,0.96, s=150, color='blue', label='Nielsen', zorder=100)
axs1[3].errorbar(37.5,0.79, yerr=0.06, xerr=None, ecolor='blue', elinewidth=5, zorder=100)
axs1[3].scatter(37.5,0.79, s=150, color='blue', zorder=100)
axs1[3].errorbar(75,0.40, yerr=0.08, xerr=None, ecolor='blue', elinewidth=4, zorder=100)
axs1[3].scatter(75,0.40, s=150, color='blue', zorder=100)
axs1[3].errorbar(150,0.25, yerr=0.11, xerr=None, ecolor='blue', elinewidth=4, zorder=100)
axs1[3].scatter(150,0.25, s=100, color='blue', zorder=100)

axs1[3].set_xlabel(r'$R_\perp$ (kpc)', fontsize=20)
axs1[3].set_xlim(0,200) 
axs1[3].set_ylim(-0.05,1.04)
axs1[3].grid(axis='both')
axs1[3].set_ylabel(r'Covering fraction', fontsize=19, labelpad=-63)
axs1[3].tick_params(axis='both', which='major', length=8, width=3, labelsize=17)
axs1[3].tick_params(axis='both', which='minor', length=4, width=2, labelsize=17)
axs1[3].legend(fontsize=18)

axs1[0].set_ylabel('Normalized Frequency', fontsize=20)
axs1[0].set_xlabel('No of cloudlets intersected', fontsize=20)
axs1[1].set_xlabel(r'log$_{10} \ (\rm N_{\rm Mg II} \ (cm^{-2}$))', fontsize=20)
axs1[2].set_xlabel(r'MgII EW ($\rm \AA$)', fontsize=20)

axs1[0].set_xscale('log')

axs1[0].set_xlim(1,max(cnt)+1)
axs1[1].set_xlim(10.5,max(col_den)+1)
axs1[2].set_xlim(0,1.4)

axs1[0].set_ylim(1.e-4,1)
axs1[1].set_ylim(1.e-4,1)
axs1[2].set_ylim(1.e-4,1)

axs1[0].grid(axis='both')
axs1[1].grid(axis='both')
axs1[2].grid(axis='both')

axs1[0].tick_params(axis='both', which='major', length=8, width=3, labelsize=17)
axs1[0].tick_params(axis='both', which='minor', length=4, width=2, labelsize=17)
axs1[1].tick_params(axis='x', which='major', length=8, width=3, labelsize=17)
axs1[1].tick_params(axis='x', which='minor', length=4, width=2, labelsize=17)
axs1[1].tick_params(axis='y', which='major', direction='out', length=8, width=3, labelsize=17)
axs1[1].tick_params(axis='y', which='minor', direction='out', length=4, width=2, labelsize=17)
axs1[2].tick_params(axis='x', which='major', length=8, width=3, labelsize=17)
axs1[2].tick_params(axis='x', which='minor', length=4, width=2, labelsize=17)  
axs1[2].tick_params(axis='y', which='major', direction='out', length=8, width=3, labelsize=17)
axs1[2].tick_params(axis='y', which='minor', direction='out', length=4, width=2, labelsize=17)

axs1[0].legend(fontsize=17)

fig1.tight_layout()
plt.subplots_adjust(wspace=0.18)

fig1.savefig("./figures/figure1_{}.pdf".format(var_name))
plt.show()  
     

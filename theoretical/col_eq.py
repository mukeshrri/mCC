# Fig 5 in paper

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

mu = 0.6
mp = 1.6726e-24
Msun = 1.989e33
kpc = 3.086e21
kB = 1.3807e-16
c = 2.9979e10
amu = 1.6737e-24
kms = 1.e5

Temp = 1.e4
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
wave0 = 2796.3553 # rest-frame transition in Angstrom
a_ion = 24.305 
mass_ion = a_ion*amu
b_dop = np.sqrt(2.*kB*Temp/mass_ion)/kms # doppler width in km/s
b_turb1 = np.sqrt(2./3.)*(1./280.)**(1./3.)*50.
b_tot1 = np.sqrt(b_dop**2. + b_turb1**2.)
b_turb2 = np.sqrt(2./3.)*(5./280.)**(1./3.)*50.
b_tot2 = np.sqrt(b_dop**2. + b_turb2**2.)
b_turb3 = np.sqrt(2./3.)*(10./280.)**(1./3.)*50.
b_tot3 = np.sqrt(b_dop**2. + b_turb3**2.)
b_turb4 = np.sqrt(2./3.)*(20./280.)**(1./3.)*50.
b_tot4 = np.sqrt(b_dop**2. + b_turb4**2.)
v_los = 0.

col = np.logspace(9,18,100)
EW0 = np.zeros(col.shape[0])
EW1 = np.zeros(col.shape[0])
EW2 = np.zeros(col.shape[0])
EW3 = np.zeros(col.shape[0])
EW4 = np.zeros(col.shape[0])

ew_cal = lambda col : 8.8 * (col/1.e12) * f_lu * (wave0/1.e3)**2. *1.e-3
ew_cal_mid = lambda col : 66.7 * (b_dop/10.) * (wave0/1.e3) * np.sqrt(np.log(2.16*f_lu*(wave0/1.e3)*(col/1.e13)*(10./b_dop)))*1.e-3
ew_cal_high = lambda col : 3.062 * np.sqrt((col/1.e14) * f_lu * (gamma_ul/1.e8)) * (wave0/1.e3)**2. * 1.e-3
col_b = np.logspace(12,19,100)
col_h = np.logspace(8,19,100)

for i in range(col.shape[0]):
    nu, tau = absorption.return_optical_depth(f_lu, col[i], gamma_ul, b_dop, wave0, v_los, 0.01)
    flux = absorption.generate_norm_flux(tau)
    wave = c/nu*1.e8
    EW0[i] = absorption.return_EW(wave,flux)
    
    nu, tau = absorption.return_optical_depth(f_lu, col[i], gamma_ul, b_tot1, wave0, v_los, 0.01)
    flux = absorption.generate_norm_flux(tau)
    wave = c/nu*1.e8
    EW1[i] = absorption.return_EW(wave,flux)
    
    nu, tau = absorption.return_optical_depth(f_lu, col[i], gamma_ul, b_tot2, wave0, v_los, 0.01)
    flux = absorption.generate_norm_flux(tau)
    wave = c/nu*1.e8
    EW2[i] = absorption.return_EW(wave,flux)
    
    nu, tau = absorption.return_optical_depth(f_lu, col[i], gamma_ul, b_tot3, wave0, v_los, 0.01)
    flux = absorption.generate_norm_flux(tau)
    wave = c/nu*1.e8
    EW3[i] = absorption.return_EW(wave,flux)
    
    nu, tau = absorption.return_optical_depth(f_lu, col[i], gamma_ul, b_tot4, wave0, v_los, 0.01)
    flux = absorption.generate_norm_flux(tau)
    wave = c/nu*1.e8
    EW4[i] = absorption.return_EW(wave,flux)

np.savetxt('col_eq_MgII.txt', np.column_stack((col,EW0,EW1,EW2,EW3,EW4)), fmt='%.4e')

col,EW0,EW1,EW2,EW3 = np.loadtxt('col_eq_MgII.txt', usecols=(0,1,2,3,4), unpack=True)

fig, axs = plt.subplots(figsize=(10,8))       
   
axs.plot(col, EW0, 'k-', lw=4, label=r'$b_{{\rm turb}}=0 \,\rm  km \, s^{-1}$')   
axs.plot(col, EW1, 'k--', lw=4, label=r'$b_{{\rm turb}}={:.1f} \, \rm km \, s^{{-1}}$'.format(b_turb1))  
axs.plot(col, EW3, 'k-.', lw=4, label=r'$b_{{\rm turb}}={:.1f}\,\rm  km \, s^{{-1}}$'.format(b_turb3))  
       
axs.plot(col[:50], ew_cal(col)[:50], 'b-', lw=4, label='optically thin')
axs.plot(col_b, ew_cal_mid(col_b), 'r-', lw=4, zorder=100, label='flat portion')
axs.plot(col_h, ew_cal_high(col_h), 'm-', lw=4, zorder=100, label='damped portion')
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(1.e11,2.e18)
axs.set_ylim(1.e-2,1.e1)
axs.set_xlabel(r'MgII column density ($\rm cm^{-2}$)', size=24)
axs.set_ylabel(r'MgII equivalent width ($\rm \AA$)', size=24)
axs.tick_params(which='major', axis='both', length=8, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=4, width=2, labelsize=20)
plt.grid()
axs.legend(ncol=2, loc='upper left', fontsize=19.8)
fig.tight_layout()
plt.savefig('./figures/col_eq_comp.pdf')
plt.show()


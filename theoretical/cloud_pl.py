# Plot column density and EW for power-law distribution 

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
from scipy import integrate
import velocity_generator as turb_vel
from scipy.optimize import minimize
import scipy.signal as signal

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

rCGM = 280
M_star = 6.08e10

Mcold = 1.e10
Ncc = 1.e3
Rcc = 10
Mcc = Mcold/Ncc

alpha = 1.2
r0 = 1.
a_l = 10
a_u = rCGM-Rcc
mg_factor = 1.7e-6

Nlos = 1.e4
density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp)
print('cold gas number density : ',density)
print('volume fraction (%) : ', Ncc*(Rcc/rCGM)**3.*1.e2)
print('area covering fraction : ', Ncc*(Rcc/rCGM)**2.)

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

r_los = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
     
fl = open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),'rb') 
data_cl = pickle.load(fl)
fl.close()  
       
for key_val in data_cl.keys():
    exec(f"{key_val} = data_cl[\"{key_val}\"]")
res = res 
     
col_den = np.zeros((r_los.shape[0]))
     
for j in range(r_los.shape[0]):
    if_tuple = isinstance(res[j], tuple)
    if if_tuple:
       arr = np.array(res[j][1])
       col_den[j] = np.sum(arr)
       
EW = np.loadtxt("./data/ew_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc))

EW_avg = []
EW_med = []
col_avg = []
col_med = []
rad_bin = np.arange(1,rCGM+1,10)
for i in range(rad_bin.shape[0]-1):
    condition = np.logical_and(r_los>=rad_bin[i], r_los<rad_bin[i+1])
    #condition = np.logical_and(condition, n_int!=0)
    EW_avg.append(np.average(EW[condition])) 
    EW_med.append(np.median(EW[condition]))   
    col_avg.append(np.average(col_den[condition]))
    col_med.append(np.median(col_den[condition])) 
    
np.savetxt('mean_EW_pl.txt', np.column_stack(((rad_bin[:-1]+rad_bin[1:])/2.,EW_avg)))  # generate this only for the fiducial run   

n0 = Ncc*(3.-alpha)/(4.*np.pi*r0**3.)/((a_u/r0)**(3.-alpha) - (a_l/r0)**(3.-alpha))
No_cc = np.zeros(r_perp.shape[0])
for i in range(r_perp.shape[0]):
    func = lambda r : r**(1.-alpha)/np.sqrt(r**2. - r_perp[i]**2.)
    No_cc[i] = integrate.quad(func, r_perp[i]+0.001, rCGM)[0]
No_cc = 2.*n0*np.pi*Rcc**2.*r0**alpha*No_cc    
col_th = No_cc*4./3.*Rcc*mg_factor*density*kpc
col_sigma = No_cc*np.sqrt(2.)/3.*Rcc*mg_factor*density*kpc

fig, axs = plt.subplots(figsize=(10,8))

axs.scatter(r_los, col_den, color='darkgray', facecolors='none', s=50, zorder=100)  
axs.plot((rad_bin[:-1]+rad_bin[1:])/2., col_avg, 'k-', linewidth=4, label='mean', zorder=100) 

axs.plot(r_perp, col_th, 'r-', linewidth=4, label='analytical', zorder=100) 
axs.plot(r_perp, col_th+col_sigma, 'r:', linewidth=4, zorder=100) 
axs.plot(r_perp, col_th-col_sigma, 'r:', linewidth=4, zorder=100)

plt.text(50,1.2e14,'power law distribution', fontsize=20., color='black')  
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(1.e11,2.e14)
axs.set_xlabel(r'impact parameter ($R_{\perp}$) [kpc]', size=20)
axs.set_ylabel(r'$N_{\rm MgII} \, [\rm cm^{-2}]$', size=20)
axs.tick_params(which='both', axis='both', length=4, width=1, labelsize=15)
plt.grid()
axs.legend(loc='lower left', fontsize=18)
fig.tight_layout()
plt.savefig('./figures/col_pl.pdf')
plt.show()

fig, axs = plt.subplots(figsize=(10,8))
axs.plot(r_perp, nielsen(r_perp), lw=4, color='blue', zorder=100, label='Nielsen')
axs.plot(r_perp, nielsen_l(r_perp), lw=4, color='blue', ls=':', zorder=100) 
axs.plot(r_perp, nielsen_u(r_perp), lw=4, color='blue', ls=':', zorder=100) 

axs.plot(r_perp, huang(r_perp), lw=4, color='green', zorder=100, label='Huang')  
axs.plot(r_perp, huang_l(r_perp), lw=4, color='green', ls=':', zorder=100)
axs.plot(r_perp, huang_u(r_perp), lw=4, color='green', ls=':', zorder=100)
 
axs.plot(r_perp, dutta(r_perp), lw=4, color='magenta', zorder=100, label='Dutta') 
axs.plot(r_perp, dutta_l(r_perp), lw=4, color='magenta', ls=':', zorder=100)
axs.plot(r_perp, dutta_u(r_perp), lw=4, color='magenta', ls=':',zorder=100) 

axs.scatter(r_los, EW, color='darkgray', facecolors='none', s=50, zorder=100) 

axs.plot((rad_bin[:-1]+rad_bin[1:])/2., EW_avg, 'k-', linewidth=4, label='mean', zorder=100)

plt.text(40,3.5,r'power law ($\alpha=1.2$) distribution', fontsize=22, color='black')
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(1.e-2, 4.5)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=22)
axs.set_ylabel(r'Equivalent width MgII $[\rm \AA]$', size=22)
axs.tick_params(which='major', axis='both', length=10, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=4, width=2, labelsize=20)
plt.grid()
axs.legend(loc='lower left', ncol=2,fontsize=22)
fig.tight_layout()
plt.savefig('./figures/EW_pl.pdf')
plt.show()


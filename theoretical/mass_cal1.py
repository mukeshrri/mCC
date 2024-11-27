# Fig 9 in paper

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
Mcold_ = np.array([1.e9,1.e10,1.e11])
M_star = 6.08e10
Ncc_ = np.array([1.e2,1.e3,1.e4])
Mcc = Mcold_/Ncc_
Rcc_ = np.array([5,10,20])
Nlos = 1.e4

r_bin = np.linspace(10,rCGM**2.,10)
r_bin = np.sqrt(r_bin)
no_pt = 10

color = ['green','deepskyblue','magenta','blue']
marker = ['<', 'o', '>']
ls = ['-', '--', ':']
fig, axs = plt.subplots(figsize=(10,8))

for l in range(Ncc_.shape[0]):
 Ncc = Ncc_[l]
 for m in range(Rcc_.shape[0]):
  Rcc = Rcc_[m]
  for i in range(Mcold_.shape[0]):
    Mcold = Mcold_[i]
    n_empty = 0
    col_tot = []
    
    r_los = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
    
    col_den = np.zeros((r_los.shape[0]))
    
    fl = open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),'rb') 
    data_cl = pickle.load(fl)
    fl.close()  
       
    for key_val in data_cl.keys():
        exec(f"{key_val} = data_cl[\"{key_val}\"]")
    res = res 
 
    for j in range(r_los.shape[0]):
        if_tuple = isinstance(res[j], tuple)
        if if_tuple:
           arr = np.array(res[j][1])
           col_den[j] = np.sum(arr)
       
    for j in range(r_bin.shape[0]-1):
        
        condition = np.logical_and(r_los>=r_bin[j], r_los<r_bin[j+1])
        
        nlos = r_los[condition].shape[0]
        
        indx = np.random.randint(low=0, high=nlos, size=no_pt)
        
        cond_empty = col_den[condition][indx] == 0
        cond_noempty = col_den[condition][indx] != 0
        n_empty += col_den[condition][indx][cond_empty].shape[0]
        _col = list(col_den[condition][indx][cond_noempty])
        col_tot = col_tot + _col
  
    col_tot = np.array(col_tot)
    avg_col = np.average(col_tot)
    err = np.std(col_tot)
    area_cov = 1. - (n_empty/(no_pt*(j+1))) 
    atom_tot = avg_col*area_cov
    err_atom = area_cov*err
   
    axs.errorbar(area_cov, avg_col, yerr=err, color='black', ecolor=color[l], elinewidth=4, capsize=10., capthick=4., linestyle='--')
    
    axs.scatter(area_cov, avg_col, marker=marker[m], color='black', facecolors=color[l], ls=ls[i], lw=3, s=600, zorder=100) 

fa = np.linspace(0.01,1.,100)
axs.plot(fa, 1.5e14/fa, color='black', ls=':', lw=5, alpha=0.5)
axs.plot(fa, 1.5e13/fa, color='black', ls='--', lw=5, alpha=0.5)
axs.plot(fa, 1.5e12/fa, color='black', ls='-', lw=5, alpha=0.5)

z,D,EW,errEW,logN,errN = np.loadtxt("coshalo_MgII.txt", skiprows=1, unpack=True)
condition1 = np.logical_and(errN <1, errN >-1)
condition = errN <1
mean_col = np.mean(10**logN[condition])
err_ind = 10**(logN[condition1]+ np.log10(1.-10**(-errN[condition1])))
err_num = np.std(10**logN[condition])
err_tot = np.sqrt(np.mean(10**(2.*np.log10(err_ind))) + err_num**2.)
fa = logN[condition].shape[0]/logN.shape[0]

axs.errorbar(fa,mean_col, yerr=err_tot, ecolor='olive', elinewidth=4, zorder=100)
axs.scatter(fa,mean_col, color='olive', s=600, zorder=100)

logN,errl,erru,D = np.loadtxt("CUBSVI_MgII.txt", skiprows=1, usecols=(1,2,3,4), unpack=True)
errN = errl
mean_col = np.mean(10**logN)
err_ind = 10**(logN+ np.log10(1.-10**(-errN)))
err_num = np.std(10**logN)
err_tot = np.sqrt(np.mean(10**(2.*np.log10(err_ind))) + err_num**2.)

axs.errorbar(1,mean_col, yerr=err_tot, ecolor='lawngreen', elinewidth=4, zorder=100)
axs.scatter(1,mean_col, color='lawngreen', s=600, zorder=100)


legend_elements = [Line2D([0],[0],marker='<',color='w', markerfacecolor='k',markersize=20, label=r'$R_{\rm cc}=5$ kpc'),
                   Line2D([0],[0],marker='o',color='w', markerfacecolor='k',markersize=20,label=r'$R_{\rm cc}=10$ kpc'),
                   Line2D([0],[0],marker='>',color='w',markerfacecolor='k',markersize=20,label=r'$R_{\rm cc}=20$ kpc'),
                   Patch(facecolor=color[0], label=r'$N_{\rm cc}=10^2$'),
                   Patch(facecolor=color[1], label=r'$N_{\rm cc}=10^3$'),
                   Patch(facecolor=color[2], label=r'$N_{\rm cc}=10^4$'),
                   Line2D([0],[0],marker='o',color='w', markerfacecolor='olive',markersize=20, label=r'COS-Halos'),
                   Line2D([0],[0],marker='o',color='w', markerfacecolor='lawngreen',markersize=20, label=r'CUBS:VI')
                   ]
                   
axs.text(5.e-2, 1.8e15, r'$10^{11}\, \rm M_\odot$', size=25, rotation=-20.)                   
axs.text(5.e-2, 1.8e14, r'$10^{10}\, \rm M_\odot$', size=25, rotation=-20.)                   
axs.text(5.e-2, 1.8e13, r'$10^{9}\, \rm M_\odot$', size=25, rotation=-20.)                   
axs.set_xlabel(r'$f_{\rm A}^{\rm cc}$', fontsize=24)
axs.set_ylabel(r'$\langle N_{\rm MgII} \rangle \, (\rm cm^{-2})$', fontsize=24)
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_ylim(2.e11,1.e16)
axs.set_xlim(1.e-2,1.1e0)
axs.tick_params(which='major', axis='both', direction='out', length=10, width=2, labelsize=18)
axs.tick_params(which='minor', axis='both', direction='out', length=4, width=1, labelsize=18)
axs.grid(axis='both', which='major')
axs.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize=19)
fig.tight_layout()
plt.savefig('./figures/col_cov_frac.pdf')
plt.show()
plt.close()


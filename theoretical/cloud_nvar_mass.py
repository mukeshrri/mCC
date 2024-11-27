# Density varying with r, f_MgII depends on denisty of cold gas at that radius (Bottom panel of Fig 16 in paper)

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

proc = 1

XH = 0.74
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
Ncc = 1.e3
Rcc = 10
Nlos = 1.e4

Mcold_ = np.array([1.e11,1.e10,1.e9])
Mcc = Mcold_/Ncc
fig, axs = plt.subplots(figsize=(10,8))
color = ['peru','darkgrey','teal']
for i in range(Mcold_.shape[0]):

 Mcold = Mcold_[i]

 X,Y,Z = np.loadtxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), unpack=True)

 r = np.sqrt(X**2. + Y**2. + Z**2.)
    
 r_perp = np.linspace(1,rCGM,100)
       
 num_cold = 0.2*(20./r_perp)**1.5

 density_ph = np.interp(r, r_perp, num_cold)

 n,frac = np.loadtxt("ion_frac_Mg_KS18_z0.3_T4.0_met0.3.txt", usecols=(0,2), unpack=True)

 n = 10**n/(mu*XH)
 n = np.sort(n)
 frac = np.sort(frac)

 mg_factor = mu*XH*0.3*3.47e-5*np.interp(np.log10(density_ph),np.log10(n),frac)
 
 density = Mcc[i]*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp) 
 
 r = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
 th = np.random.uniform(low=0, high=2.*np.pi, size=int(Nlos))
 xlos = r*np.cos(th)
 ylos = r*np.sin(th)
  
 def cloud_intersect(data_cl):
    xl = data_cl[0]
    yl = data_cl[1]
    det = Rcc**2. - (xl-X)**2. - (yl-Y)**2.
    condition = det>=0.
    cl_indx = cloud_index[condition]
    det = det[condition]
    if len(det)==0:
       return 0
    else :
      c_len = 2.*np.sqrt(det)
      c_den = c_len*density*mg_factor[condition]*kpc
      
      return cl_indx,c_den

 cloud_index = np.arange(0,int(Ncc),1)
 data_cl = np.column_stack((xlos,ylos))
 with multiprocessing.Pool(processes=proc) as pool:
     res = pool.map(cloud_intersect,data_cl)
             
 print('complete! \n')
 print("--- %s seconds ---\n " % (time.time() - start_time)) 

 r_los = np.sqrt(xlos**2. + ylos**2.)
 col_den = np.zeros(xlos.shape[0])
 n_int = np.zeros(xlos.shape[0])

 for j in range(xlos.shape[0]):
   if_tuple = isinstance(res[j], tuple)
   if if_tuple:
       n_int[j] = np.array(res[j][0]).shape[0]
       arr = np.array(res[j][1])
       arr0 = np.array(res[j][0])
       col_den[j] = np.sum(arr)
 
 condition = col_den == 0
 cov_frac = (Nlos-col_den[condition].shape[0])/Nlos
 
 #print('covering fraction is :', cov_frac)
 
 col_avg = []
 rad_bin = np.arange(0,rCGM+1,20)
 for j in range(rad_bin.shape[0]-1):
    condition = np.logical_and(r_los>rad_bin[j], r_los<=rad_bin[j+1])
    #condition = np.logical_and(condition,col_den!=0)
    col_avg.append(np.average(col_den[condition]))   

 if i ==1:
    axs.scatter(r_los, col_den, color=color[i], facecolors='none', s=100, alpha=1, zorder=100) 
 
 else:
   axs.scatter(r_los, col_den, color=color[i], facecolors='none', s=100, alpha=1, zorder=100)     

 axs.plot((rad_bin[:-1]+rad_bin[1:])/2., col_avg, color='black', linewidth=6, zorder=100)
 axs.plot((rad_bin[:-1]+rad_bin[1:])/2., col_avg, color=color[i], linewidth=4, zorder=100, label=r'$10^{{{:.0f}}} \, \rm M_\odot$'.format(np.log10(Mcold)))  

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(1.e11,3.e15)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=22)
axs.set_ylabel(r'$N_{\rm MgII} \, [\rm cm^{-2}]$', size=22)
axs.tick_params(which='major', axis='both', length=10, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=6, width=2, labelsize=20)
plt.grid()
axs.legend(loc='upper right', fontsize=20)
fig.tight_layout()
plt.savefig('./figures/col_p_nvar_mvar.pdf')
plt.show()
plt.close()


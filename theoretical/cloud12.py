# Fig 4 in paper

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from scipy import integrate
import pickle

rCGM= 280
Mcold = 1.e10
Ncc = 1.e3
Rcc = 10
Nlos = 1.e4

r_los = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
r_u = r_los
r_p = r_los
     
fl = open("./data/cden_u_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),'rb') 
data_cl = pickle.load(fl)
fl.close()  
       
for key_val in data_cl.keys():
    exec(f"{key_val} = data_cl[\"{key_val}\"]")
res = res 
     
colu = np.zeros((r_los.shape[0]))
     
for j in range(r_los.shape[0]):
    if_tuple = isinstance(res[j], tuple)
    if if_tuple:
       arr = np.array(res[j][1])
       colu[j] = np.sum(arr)
       
fl = open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),'rb') 
data_cl = pickle.load(fl)
fl.close()  
       
for key_val in data_cl.keys():
    exec(f"{key_val} = data_cl[\"{key_val}\"]")
res = res 
     
colp = np.zeros((r_los.shape[0]))
     
for j in range(r_los.shape[0]):
    if_tuple = isinstance(res[j], tuple)
    if if_tuple:
       arr = np.array(res[j][1])
       colp[j] = np.sum(arr)
                          
r_bin = np.linspace(10,rCGM**2.,11)
r_bin = np.sqrt(r_bin)

no_pt = 10

color = ['green','deepskyblue','magenta','maroon','olive','blue','peru','orange',
'lightcoral','lawngreen','plum','tan','aliceblue','gray']

fig, axs = plt.subplots(figsize=(10,8))

z,D,EW,errEW,logN,errN = np.loadtxt("coshalo_MgII.txt", skiprows=1, unpack=True)
condition = np.logical_and(errN <1, errN >-1)
axs.scatter(D[condition], 10**logN[condition], s=200, color=color[4], label='COS-Halos')
axs.errorbar(D[condition], 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[4], elinewidth=4)

condition = errN ==-1
axs.scatter(D[condition], 10**logN[condition], marker='^', s=200, color=color[4])

condition = errN ==1
axs.scatter(D[condition], 10**logN[condition], marker='v', s=200, color=color[4])

D,logN,errN = np.loadtxt("Paper4Table1.txt", skiprows=2, usecols=(5,15,16), unpack=True)

#condition = np.logical_and(errN > 0.0, errN<1.)
condition = errN > 0.0
axs.scatter(D[condition]*rCGM, 10**logN[condition], s=200, color=color[6], label='MAGIICAT')
axs.errorbar(D[condition]*rCGM, 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[6], elinewidth=4)

logN,errl,erru,D = np.loadtxt("CUBSVI_MgII.txt", skiprows=1, usecols=(1,2,3,4), unpack=True)
axs.scatter(D*rCGM, 10**logN, s=200, color=color[9], label='CUBS:VI', zorder=100)
axs.errorbar(D*rCGM, 10**logN, yerr=np.array([10**(logN+np.log10(1.-10**(-errl))),10**(logN+np.log10(1.-10**(-erru)))]), linestyle='None', ecolor=color[9], elinewidth=4, zorder=100)

for i in range(r_bin.shape[0]-1):
    condition_u = np.logical_and(r_u>=r_bin[i],r_u<r_bin[i+1])
    condition_p = np.logical_and(r_p>=r_bin[i],r_p<r_bin[i+1])
    
    r_u_no = r_u[condition_u].shape[0]
    r_p_no = r_p[condition_p].shape[0]
    indx_u = np.random.randint(low=0, high=r_u_no, size=no_pt)
    indx_p = np.random.randint(low=0, high=r_p_no, size=no_pt)
    
    condition0_u = colu[condition_u][indx_u] == 0.
    condition0_p = colp[condition_p][indx_p] == 0.
    
    r_u0 = r_u[condition_u][indx_u][condition0_u]
    col_u0 = 3.e11*np.ones(r_u0.shape[0])
    
    r_p0 = r_p[condition_p][indx_p][condition0_p]
    col_p0 = 3.e11*np.ones(r_p0.shape[0])
    
    if i ==0:
       axs.scatter(r_u[condition_u][indx_u], colu[condition_u][indx_u], marker='s', s=200, edgecolors=color[1], color='None', lw=3,  label='uniform')
       axs.scatter(r_p[condition_p][indx_p], colp[condition_p][indx_p], marker='D', s=200, edgecolors=color[10], color='None', lw=3, label=r'power-law,$\alpha=1.2$') 
    else :   
       axs.scatter(r_u[condition_u][indx_u], colu[condition_u][indx_u], marker='s', s=200, edgecolors=color[1], color='None', lw=3)
       axs.scatter(r_p[condition_p][indx_p], colp[condition_p][indx_p], marker='D', s=200, edgecolors=color[10], color='None', lw=3)
    
    axs.scatter(r_u0, col_u0, marker='v', s=200, edgecolors='black', color=color[1], lw=1)
    axs.scatter(r_p0, col_p0, marker='v', s=200, edgecolors='black', color=color[10], lw=1)
    
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(5.e10,2.e14)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=24)
axs.set_ylabel(r'$N_{\rm MgII} \, [\rm cm^{-2}]$', size=24)
axs.tick_params(which='major', axis='both', length=10, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=4, width=2, labelsize=20)
plt.grid()
axs.legend(ncol=3, loc='lower center', fontsize=18.7)
fig.tight_layout()
plt.savefig('./figures/col_mist_scatter.pdf')
plt.show()    

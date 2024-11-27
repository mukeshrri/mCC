# Density varying with r, f_MgII depends on denisty of cold gas at that radius (Top panel of Fig 16 in paper)

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
Mcold = 1.e10
M_star = 6.08e10
Ncc = 1.e3
Mcc = Mcold/Ncc
Rcc = 10
Nlos = 1.e4

Temp = 1.e4
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
wave0 = 2796.3553 # rest-frame transition in Angstrom
a_ion = 24.305 
mass_ion = a_ion*amu

nielsen = lambda r : np.exp(0.27 - 0.015*r)
nielsen_l = lambda r : np.exp(0.27-0.11 + (-0.015-0.002)*r)
nielsen_u = lambda r : np.exp(0.27+0.11 + (-0.015+0.002)*r)

huang = lambda r : np.exp(1.35 - 1.05*np.log10(r) + 0.21*(np.log10(M_star) - 10.3))
huang_l = lambda r : np.exp(1.35-0.25 - (1.05+0.17)*np.log10(r) + (0.21-0.08)*(np.log10(M_star) - 10.3))
huang_u = lambda r : np.exp(1.35+0.25 - (1.05-0.17)*np.log10(r) + (0.21+0.08)*(np.log10(M_star) - 10.3))

dutta = lambda r : np.exp(-0.61 - 0.008*r + 0.53*(np.log10(M_star) - 9.3))
dutta_l = lambda r : np.exp(-0.61-0.58 - (0.008+0.003)*r + (0.53-0.28)*(np.log10(M_star) - 9.3))
dutta_u = lambda r : np.exp(-0.61+0.63 - (0.008-0.004)*r + (0.53+0.34)*(np.log10(M_star) - 9.3))

X,Y,Z = np.loadtxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), unpack=True)

r = np.sqrt(X**2. + Y**2. + Z**2.)

r_perp = np.linspace(1,rCGM,100)

num_cold = 0.2*(20./r_perp)**1.5

density_ph = np.interp(r, r_perp, num_cold)

n,frac = np.loadtxt("ion_frac_Mg_KS18_z0.3_T4.0_met0.3.txt", usecols=(0,2), unpack=True)

n = 10**n/(mu*XH)
n = np.sort(n)
frac = np.sort(frac)

density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp) 

mg_factor = mu*XH*0.3*3.47e-5*np.interp(np.log10(density_ph),np.log10(n),frac)
mg_factor_flat = mu*XH*0.3*3.47e-5*0.33

b_dop = np.sqrt(2*kB*Temp/mass_ion)/kms # doppler width in km/s
b_turb = np.sqrt(2./3.)*(Rcc/rCGM)**(1./3.)*50.
b_tot = np.sqrt(b_dop**2. + b_turb**2.)

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
      c_den_flat = c_len*density*mg_factor_flat*kpc
      
      return cl_indx,c_den,c_den_flat

cloud_index = np.arange(0,int(Ncc),1)
data_cl = np.column_stack((xlos,ylos))
with multiprocessing.Pool(processes=proc) as pool:
     res = pool.map(cloud_intersect,data_cl)
             
print('complete! \n')
print("--- %s seconds ---\n " % (time.time() - start_time)) 

data_dump =  {"res": res}
    
with open("./data/cden_pl_nvar_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), 'wb') as fl:
     pickle.dump(data_dump, fl)
          
r_los = np.sqrt(xlos**2. + ylos**2.)
col_den = np.zeros(xlos.shape[0])
col_den_flat = np.zeros(xlos.shape[0])
E_width = np.zeros(xlos.shape[0])
n_int = np.zeros(xlos.shape[0])

for i in range(xlos.shape[0]):
   if_tuple = isinstance(res[i], tuple)
   if if_tuple:
       n_int[i] = np.array(res[i][0]).shape[0]
       arr = np.array(res[i][1])
       arr1 = np.array(res[i][2])
       arr0 = np.array(res[i][0])
       col_den[i] = np.sum(arr)
       col_den_flat[i] = np.sum(arr1)

col_avg = []
rad_bin = np.arange(0,rCGM+1,20)
for i in range(rad_bin.shape[0]-1):
    condition = np.logical_and(r_los>rad_bin[i], r_los<=rad_bin[i+1])
    col_avg.append(np.average(col_den_flat[condition]))   
    
color = ['green','deepskyblue','magenta','maroon','olive','blue','peru','orange',
'lightcoral','lawngreen','plum','tan','aliceblue','gray','coral','khaki','tomato',
'wheat','gold','lightpink','lightcyan','linen']

fig, axs = plt.subplots(figsize=(10,8))

axs.scatter(r_los, col_den, color='darkgray', facecolors='none', s=100)  
axs.plot((rad_bin[:-1]+rad_bin[1:])/2., col_avg, 'k-', linewidth=4, zorder=100) 

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
 
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(1.e11,1.e15)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=22)
axs.set_ylabel(r'$N_{\rm MgII} \, [\rm cm^{-2}]$', size=22)
axs.tick_params(which='major', axis='both', length=10, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=6, width=2, labelsize=20)
plt.grid()
axs.legend(loc='upper right', fontsize=22)
fig.tight_layout()
plt.savefig('./figures/col_p_nvar.pdf')
plt.show()
plt.close()

#################################
EW = np.loadtxt('./data/ew_nvar_10_3_10.txt')

EW_avg = []
rad_bin = np.arange(0,rCGM+1,20)
for i in range(rad_bin.shape[0]-1):
    condition = np.logical_and(r_los>rad_bin[i], r_los<=rad_bin[i+1])
    EW_avg.append(np.average(EW[condition])) 
    
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

axs.scatter(r_los, EW, color='darkgray', facecolors='none', s=100, zorder=100) 
axs.plot((rad_bin[:-1]+rad_bin[1:])/2., EW_avg, 'k-', linewidth=4, label='mean', zorder=100)

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(1.e-2, 7.)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=22)
axs.set_ylabel(r'Equivalent width MgII $[\rm \AA]$', size=22)
axs.tick_params(which='major', axis='both', length=8, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=4, width=2, labelsize=20)
plt.grid()
axs.legend(ncol=2, loc='lower left', fontsize=22)
fig.tight_layout()
plt.savefig('./figures/EW_p_nvar.pdf')
plt.show()


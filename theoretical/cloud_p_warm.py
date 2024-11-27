# Figures 17 in paper (warm CC)

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

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

proc = 1

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
f = 1.
Mccw = Mcc
Rccw = Rcc
Nlos = 1.e4
density = Mccw*Msun/(4./3.*np.pi*(Rccw*kpc)**3.*mu*mp)
print('cc size is (kpc) : ', Rccw)
print('cold gas number density : ',density)
print('volume fraction (%) : ', Ncc*(Rccw/rCGM)**3.*1.e2)
print('area covering fraction : ', Ncc*(Rccw/rCGM)**2.)

alpha = 1.2
r0 = 1.
a_l = 5
a_u = rCGM

ox_factor = 2.399e-01*6.5268e-5
Temp = 10.**5.5
f_lu = 0.133 # oscillator strength
gamma_ul = 4.17e+08 # probability coefficients
wave0 = 1031.9261 # rest-frame transition in Angstrom
a_ion = 15.999
mass_ion = a_ion*amu
b_dop = np.sqrt(2*kB*Temp/mass_ion)/kms # doppler width in km/s
b_turb = np.sqrt(2./3.)*(Rccw/rCGM)**(1./3.)*50.
print('b_turb : ', b_turb)
b_tot = np.sqrt(b_dop**2. + b_turb**2.)
    
color = ['green','deepskyblue','magenta','maroon','olive','blue','peru','orange',
'lightcoral','lawngreen','plum','tan','aliceblue','gray','coral','khaki','tomato',
'wheat','gold','lightpink','lightcyan','linen']

X,Y,Z = np.loadtxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc), unpack=True)

r = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
th = np.random.uniform(low=0, high=2.*np.pi, size=int(Nlos))
xlos = r*np.cos(th)
ylos = r*np.sin(th)
     
def cloud_intersect(data_cl):
    xl = data_cl[0]
    yl = data_cl[1]
    det = Rccw**2. - (xl-X)**2. - (yl-Y)**2.
    condition = det>=0.
    cl_indx = cloud_index[condition]
    det = det[condition]
    if len(det)==0:
       return 0
    else :
      c_len = 2.*np.sqrt(det)
      c_den = c_len*density*ox_factor*kpc
     
      return cl_indx,c_den

cloud_index = np.arange(0,int(Ncc),1)
data_cl = np.column_stack((xlos,ylos))
with multiprocessing.Pool(processes=proc) as pool:
     res = pool.map(cloud_intersect,data_cl)
             
print('complete! \n')
print("--- %s seconds ---\n " % (time.time() - start_time)) 

r_los = np.sqrt(xlos**2. + ylos**2.)
col_den = np.zeros(xlos.shape[0])
E_width = np.zeros(xlos.shape[0])
n_int = np.zeros(xlos.shape[0])

for i in range(xlos.shape[0]):
   if_tuple = isinstance(res[i], tuple)
   if if_tuple:
       n_int[i] = np.array(res[i][0]).shape[0]
       arr = np.array(res[i][1])
       col_den[i] = np.sum(arr)

condition = (col_den == 0.)
high = r_los[condition].shape[0]
no_empty = 10
EW_min = np.zeros(no_empty)
Col_min = np.zeros(no_empty)
r_los_0 = np.zeros(no_empty)
for k in range(no_empty):
    indx = np.random.randint(low=0, high=high, size=1)[0]
    r_los_0[k] = r_los[condition][indx]
    EW_min[k] = 2.e-2
    Col_min[k] = 7.e12
         
r_perp = np.linspace(1,rCGM,100)
n0 = Ncc*(3.-alpha)/(4.*np.pi*r0**3.)/((a_u/r0)**(3.-alpha) - (a_l/r0)**(3.-alpha))
No_cc = np.zeros(r_perp.shape[0])
for i in range(r_perp.shape[0]):
    func = lambda r : r**(1.-alpha)/np.sqrt(r**2. - r_perp[i]**2.)
    No_cc[i] = integrate.quad(func, r_perp[i]+0.001, rCGM)[0]
No_cc = 2.*n0*np.pi*Rccw**2.*r0**alpha*No_cc    
col_th = No_cc*4./3.*Rccw*ox_factor*density*kpc
col_sigma = No_cc*np.sqrt(2.)/3.*Rccw*ox_factor*density*kpc

fig, axs = plt.subplots(figsize=(10,8))

axs.scatter(r_los, col_den, color='darkgray', facecolors='none', s=100)  

D,logN,errN = np.loadtxt("coshalo_OVI.txt", skiprows=1, usecols=(1,5,6), unpack=True)
condition = np.logical_and(errN <1, errN >-1)
axs.scatter(D[condition], 10**logN[condition], s=200, color=color[4], label='COS-Halos')
axs.errorbar(D[condition], 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[4], elinewidth=4)

condition = errN ==-1
axs.scatter(D[condition], 10**logN[condition], marker='^', s=200, color=color[4])

condition = errN ==1
axs.scatter(D[condition], 10**logN[condition], marker='v', s=200, color=color[4])

D,logN,errl,erru = np.loadtxt("CUBS24_OVI.txt", skiprows=2, usecols=(9,11,12,13), unpack=True)
condition = np.logical_and(D<1.,errl<1.)
axs.scatter(D[condition]*rCGM, 10**logN[condition], s=200, color=color[9], label='CUBS:VII')
axs.errorbar(D[condition]*rCGM, 10**logN[condition], yerr=np.array([10**(logN[condition]+np.log10(1.-10**(-errl[condition]))),10**(logN[condition]+np.log10(1.-10**(-erru[condition])))]), linestyle='None', ecolor=color[9], elinewidth=4)

condition = np.logical_and(D<1.,errl==1)
axs.scatter(D[condition]*rCGM, 10**logN[condition], marker='v', s=200, color=color[9])

D,r200,logN,flag = np.loadtxt("CGM2_OVI_reduced.txt", skiprows=1, usecols=(1,4,5,6), unpack=True)
d = D/r200
condition = np.logical_and(flag ==0, d<1.)
axs.scatter(d[condition]*rCGM, 10**logN[condition], marker='o', s=200, color=color[14], label=r'CGM$^2$')

condition = np.logical_and(flag ==1, d<1.)
axs.scatter(d[condition]*rCGM, 10**logN[condition], marker='v', s=200, color=color[14])

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(3.e12,2.e15)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=22)
axs.set_ylabel(r'$N_{\rm OVI} \, [\rm cm^{-2}]$', size=22)
axs.tick_params(which='major', axis='both', length=10, width=3, labelsize=18)
axs.tick_params(which='minor', axis='both', length=6, width=2, labelsize=18)
plt.grid()
axs.legend(loc='lower left', fontsize=22)
fig.tight_layout()
plt.savefig('./figures/col_OVI_pl.pdf')
plt.show()


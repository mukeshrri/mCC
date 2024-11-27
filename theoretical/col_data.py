# Figure 3 in paper

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from scipy import integrate

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
Mcold = 1.e10
M_star = 6.08e10
Ncc = 1.e3
Mcc = Mcold/Ncc
Rcc = 10

alpha = 1.2
r0 = 1.
a_l = 10
a_u = rCGM

density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp)
print('cold gas number density : ',density)
print('volume fraction is (%) : ', Ncc*(Rcc/rCGM)**3.*1e2)
print('area covering fraction is : ', Ncc*(Rcc/rCGM)**2.)

mg_factor = 1.7e-6

r_perp = np.linspace(1,rCGM,50)

den_CGM = Mcold*Msun/(4./3.*np.pi*(rCGM*kpc)**3.) / (mu*mp)
col_CGM = den_CGM*mg_factor*2.*np.sqrt(rCGM**2. - r_perp**2.)*kpc

No_cc_u = 3.*Ncc/(2.*rCGM**3.)*Rcc**2.*r_perp*np.sqrt((rCGM/r_perp)**2. - 1.)
col_th_u = No_cc_u *4./3.*Rcc *mg_factor*density*kpc
avg_L = 4./3. * Rcc*kpc
sig_Ncc = np.sqrt(No_cc_u)
sig_L = np.sqrt(2.)/3. * Rcc*kpc
col_sigma_u = (sig_Ncc*avg_L)**2. + (No_cc_u*sig_L)**2. + (sig_L*sig_Ncc)**2. 
col_sigma_u = np.sqrt(col_sigma_u)*mg_factor*density

n0 = Ncc*(3.-alpha)/(4.*np.pi*r0**3.)/((a_u/r0)**(3.-alpha) - (a_l/r0)**(3.-alpha))
No_cc_p = np.zeros(r_perp.shape[0]-1)
for i in range(r_perp.shape[0]-1):
    func = lambda r : r**(1.-alpha)/np.sqrt(r**2. - r_perp[i]**2.)
    No_cc_p[i] = integrate.quad(func, r_perp[i]+0.0001, rCGM)[0]

No_cc_p = 2.*n0*np.pi*Rcc**2.*r0**alpha*No_cc_p    
col_th_p = No_cc_p*4./3.*Rcc*mg_factor*density*kpc

avg_L = 4./3. * Rcc*kpc
sig_Ncc = np.sqrt(No_cc_p)
sig_L = np.sqrt(2.)/3. * Rcc*kpc
col_sigma_p = (sig_Ncc*avg_L)**2. + (No_cc_p*sig_L)**2. + (sig_L*sig_Ncc)**2. 
col_sigma_p = np.sqrt(col_sigma_p)*mg_factor*density

color = ['green','deepskyblue','magenta','maroon','olive','blue','peru','orange',
'lightcoral','lawngreen','plum','tan','aliceblue','gray','coral','khaki','tomato',
'wheat','gold','lightpink','lightcyan','linen']


fig, axs = plt.subplots(figsize=(10,8))

axs.plot(r_perp, col_th_u, color=color[1], ls='-', lw=5, label='uniform', zorder=100) 
axs.plot(r_perp, col_th_u+col_sigma_u, color=color[1], ls=':', lw=5, zorder=100) 
axs.plot(r_perp, col_th_u-col_sigma_u, color=color[1], ls=':', lw=5, zorder=100)

axs.plot(r_perp[:-1], col_th_p, color=color[2], ls='-', lw=5, label=r'power-law,$\alpha=1.2$', zorder=100)  
axs.plot(r_perp[:-1], col_th_p+col_sigma_p, color=color[2], ls=':', lw=5, zorder=100) 
axs.plot(r_perp[:-1], col_th_p-col_sigma_p, color=color[2], ls=':', lw=5, zorder=100)

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

D,N = np.loadtxt("column_MgII.txt", unpack=True)
axs.plot(D*rCGM, 10**N, ls='-', lw=5, color='black', label='Dutta2024')

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,300)
axs.set_ylim(9.e10,2.e14)
axs.set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=24)
axs.set_ylabel(r'$N_{\rm MgII} \, [\rm cm^{-2}]$', size=24)
axs.tick_params(which='major', axis='both', length=10, width=3, labelsize=20)
axs.tick_params(which='minor', axis='both', length=4, width=2, labelsize=20)
plt.grid()
axs.legend(ncol=2, loc='lower left', fontsize=20)
fig.tight_layout()
plt.savefig('./figures/col_mist.pdf')
plt.show()

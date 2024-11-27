# Fig 6 in paper

import numpy as np
import matplotlib.pyplot as plt
import absorption_calculator as absorption
import multiprocessing

proc = 7

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
Temp = 1.e4
f_lu = 0.6155 # oscillator strength
gamma_ul = 2.625e+08 # probability coefficients
wave0 = 2796.3553 # rest-frame transition in Angstrom
a_ion = 24.305 
mass_ion = a_ion*amu
b_dop = np.sqrt(2.*kB*Temp/mass_ion)/kms # doppler width in km/s
v_los = 0.
v_max = 50.

Rcc = np.logspace(0,1.5,30)
Ncc = np.logspace(2,6,30)

Rcc,Ncc = np.meshgrid(Rcc,Ncc)

def col(Ncc,Rcc):
    Mcc = Mcold/Ncc
    density = Mcc*Msun/(4./3.*np.pi*(Rcc*kpc)**3.*mu*mp)
    return 1.7e-6 * density * 2.* Rcc * kpc  
    
col_den = col(Ncc,Rcc)

def cal_EW(data_cloud):
    rcc = data_cloud[0]
    col_d = data_cloud[1]
    b_turb = np.sqrt(2./3.)*(rcc/rCGM)**(1./3.) * v_max
    b_tot = np.sqrt(b_dop**2. + b_turb**2.)
    nu, tau = absorption.return_optical_depth(f_lu, col_d, gamma_ul, b_tot, wave0, v_los, 0.01)
    flux = absorption.generate_norm_flux(tau)
    wave = c/nu*1.e8
    eq_wd = absorption.return_EW(wave,flux)
    return eq_wd
    
data_cl = np.column_stack((Rcc.flatten(),col_den.flatten()))
with multiprocessing.Pool(processes=proc) as pool:
     res = pool.map(cal_EW,data_cl)

eq_wd = np.reshape(res, Rcc.shape)  
  
fig, ax = plt.subplots(figsize=(10,8))
cs = ax.contour(Ncc,Rcc, np.log10(col_den), levels=[10,11], colors='black', linewidths=2)
cs0 = ax.contour(Ncc,Rcc, np.log10(col_den), levels=[12,13,14,15], colors='black', linewidths=2)
cs1 = ax.contour(Ncc,Rcc, eq_wd, levels=[0.001], colors='magenta', linewidths=2)
cs2 = ax.contour(Ncc,Rcc, eq_wd, levels=[0.01], colors='magenta', linewidths=2)
cs3 = ax.contour(Ncc,Rcc, eq_wd, levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7], colors='magenta', linewidths=2)

ax.clabel(cs, fontsize=22, inline=10, rightside_up=False, inline_spacing=25, use_clabeltext=True)
ax.clabel(cs0, fontsize=22, inline=10, rightside_up=False, inline_spacing=-10, use_clabeltext=True)
ax.clabel(cs1, fontsize=22, inline=10, rightside_up=False, inline_spacing=40, use_clabeltext=True)
ax.clabel(cs2, fontsize=22, inline=10, rightside_up=False, inline_spacing=0, use_clabeltext=True)
ax.clabel(cs3, fontsize=22, inline=10, rightside_up=False, inline_spacing=-8, use_clabeltext=True)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$N_{\rm cc}$', fontsize=24)
ax.set_ylabel(r'$R_{\rm cc}$ [kpc]', fontsize=24)
ax.tick_params(which='major', axis='both', length=8, width=3, labelsize=20)
ax.tick_params(which='minor', axis='both', length=6, width=2, labelsize=20)
ax.grid()
plt.tight_layout()
plt.savefig('./figures/contour.pdf')
plt.show()
    

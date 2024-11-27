# Fig 8 in paper

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
Nlos = 1.e4

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

colors= 'red'
color = ['green','deepskyblue','magenta','maroon','olive','blue','peru','orange',
'lightcoral','lawngreen','plum','tan','aliceblue','gray','coral','khaki','tomato',
'wheat','gold','lightpink','lightcyan','linen']

############################################################## variation with Mcold

Mcold = np.array([1.e9,1.e10,1.e11])
Ncc = 1.e3
Mcc = Mcold/Ncc
Rcc = 10

fig1, axs1 = plt.subplots(nrows=1,ncols=3,figsize=(14,4))
fig2, axs2 = plt.subplots(nrows=1,ncols=3,figsize=(14,4))
lw = 3
     
for i in range(Mcold.shape[0]):
     
     r_los = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
     
     fl = open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold[i])),int(np.log10(Ncc)),Rcc),'rb') 
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
     
     EW = np.loadtxt("./data/ew_pl_{}_{}_{}.txt".format(int(np.log10(Mcold[i])),int(np.log10(Ncc)),Rcc))
     
     #################################
     axs1[i].scatter(r_los, EW, color='darkgray', facecolors='none', s=50) 
     
     axs1[i].plot(r_perp, nielsen(r_perp), lw=lw, color='blue', zorder=100, label='Nielsen')
     axs1[i].plot(r_perp, nielsen_l(r_perp), lw=lw, color='blue', ls=':', zorder=100) 
     axs1[i].plot(r_perp, nielsen_u(r_perp), lw=lw, color='blue', ls=':', zorder=100) 

     axs1[i].plot(r_perp, huang(r_perp), lw=lw, color='green', zorder=100, label='Huang')  
     axs1[i].plot(r_perp, huang_l(r_perp), lw=lw, color='green', ls=':', zorder=100)
     axs1[i].plot(r_perp, huang_u(r_perp), lw=lw, color='green', ls=':', zorder=100)
    
     axs1[i].plot(r_perp, dutta(r_perp), lw=lw, color='magenta', zorder=100, label='Dutta') 
     axs1[i].plot(r_perp, dutta_l(r_perp), lw=lw, color='magenta', ls=':', zorder=100)
     axs1[i].plot(r_perp, dutta_u(r_perp), lw=lw, color='magenta', ls=':',zorder=100)
    
     ####################################
     axs2[i].scatter(r_los, col_den, color='darkgray', facecolors='none', s=50)
     
     z,D,EW,errEW,logN,errN = np.loadtxt("coshalo_MgII.txt", skiprows=1, unpack=True)
     condition = np.logical_and(errN <1, errN >-1)
     axs2[i].scatter(D[condition], 10**logN[condition], s=50, color=color[4], label='COS-Halos', zorder=100)
     axs2[i].errorbar(D[condition], 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[4], elinewidth=2, zorder=100)

     condition = errN ==-1
     axs2[i].scatter(D[condition], 10**logN[condition], marker='^', s=50, color=color[4], zorder=100)

     condition = errN ==1
     axs2[i].scatter(D[condition], 10**logN[condition], marker='v', s=50, color=color[4], zorder=100)

     D,logN,errN = np.loadtxt("Paper4Table1.txt", skiprows=2, usecols=(5,15,16), unpack=True)
     #condition = np.logical_and(errN > 0.0, errN<1.)
     condition = errN > 0.0
     axs2[i].scatter(D[condition]*rCGM, 10**logN[condition], s=50, color=color[6], label='MAGIICAT', zorder=100)
     axs2[i].errorbar(D[condition]*rCGM, 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[6], elinewidth=2, zorder=100)
     
     logN,errl,erru,D = np.loadtxt("CUBSVI_MgII.txt", skiprows=1, usecols=(1,2,3,4), unpack=True)
     axs2[i].scatter(D*rCGM, 10**logN, s=50, color=color[9], label='CUBS:VI', zorder=100)
     axs2[i].errorbar(D*rCGM, 10**logN, yerr=np.array([10**(logN+np.log10(1.-10**(-errl))),10**(logN+np.log10(1.-10**(-erru)))]), linestyle='None', ecolor=color[9], elinewidth=2, zorder=100)
     
     axs1[i].set_yscale('log')
     axs2[i].set_yscale('log')
     axs1[i].set_xscale('log')
     axs2[i].set_xscale('log')
     axs1[i].set_xlim(10,300)
     axs2[i].set_xlim(10,300)
     axs1[i].set_ylim(1.e-2, 2.e1)
     axs2[i].set_ylim(1.e11, 3.e15)
     if i !=0 :
        axs1[i].set_yticklabels(labels=[], minor=False)
        axs2[i].set_yticklabels(labels=[], minor=False)
        
     axs1[i].tick_params(which='major', axis='both', direction='inout', length=8, width=2, labelsize=12)
     axs1[i].tick_params(which='minor', axis='both', direction='inout', length=4, width=1, labelsize=12)
     axs2[i].tick_params(which='major', axis='both', direction='inout', length=8, width=2, labelsize=12)
     axs2[i].tick_params(which='minor', axis='both', direction='inout', length=4, width=1, labelsize=12)
     axs1[i].grid(axis='both', which='major')
     axs2[i].grid(axis='both', which='major')
     
axs1[1].set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=16)
axs2[1].set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=16)
axs1[0].set_ylabel(r'EW MgII $[\rm \AA]$', size=16)
axs2[0].set_ylabel(r'$ \rm N_{MgII} \, [\rm cm^{-2}]$', size=16)
axs1[0].legend(loc='upper left', ncol=3, fontsize=13)
axs2[0].legend(loc='upper left', ncol=1, fontsize=13)

fontsize = 17

fig1.text(0.05,0.95,r"$M_{\rm cold}  \rightarrow$", fontsize=fontsize, color=colors)
fig2.text(0.05,0.95,r"$M_{\rm cold} \rightarrow$", fontsize=fontsize, color=colors)

axs1[0].set_title(r'$10^9 \, \rm M_\odot$', color=colors, fontsize=fontsize)
axs1[1].set_title(r'$10^{10} \, \rm M_\odot$', color=colors, fontsize=fontsize)
axs1[2].set_title(r'$10^{11} \, \rm M_\odot$', color=colors, fontsize=fontsize)

axs2[0].set_title(r'$10^9 \, \rm M_\odot$', color=colors, fontsize=fontsize)
axs2[1].set_title(r'$10^{10} \, \rm M_\odot$', color=colors, fontsize=fontsize)
axs2[2].set_title(r'$10^{11} \, \rm M_\odot$', color=colors, fontsize=fontsize)

fig1.subplots_adjust(left= 0.06, bottom = 0.15, right=0.995,top=0.92,wspace=0,hspace=0)
fig2.subplots_adjust(left= 0.06, bottom = 0.15, right=0.995,top=0.92,wspace=0,hspace=0)

fig1.savefig('./figures/EW_all_p_Mvar.pdf')
#fig1.savefig('./figures/EW_all_p_Mvar.png')
fig2.savefig('./figures/Col_all_p_Mvar.pdf')
#fig2.savefig('./figures/Col_all_p_Mvar.png')
plt.show()
plt.close()

########################################### variation with Ncc,Rcc

Mcold = 1.e10
Ncc = np.array([1.e2,1.e3,1.e4])
Mcc = Mcold/Ncc
Rcc = np.array([5,10,20])

fig1, axs1 = plt.subplots(nrows=3,ncols=3,figsize=(14,10))
fig2, axs2 = plt.subplots(nrows=3,ncols=3,figsize=(14,10))
lw = 3

for i in range(Ncc.shape[0]):
  for j in range(Rcc.shape[0]):
  
     r_los = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))
     
     fl = open("./data/cden_pl_{}_{}_{}.pickle".format(int(np.log10(Mcold)),int(np.log10(Ncc[i])),Rcc[j]),'rb') 
     data_cl = pickle.load(fl)
     fl.close()  
       
     for key_val in data_cl.keys():
           exec(f"{key_val} = data_cl[\"{key_val}\"]")
     res = res 
     
     col_den = np.zeros((r_los.shape[0]))
     
     for k in range(r_los.shape[0]):
         if_tuple = isinstance(res[k], tuple)
         if if_tuple:
            arr = np.array(res[k][1])
            col_den[k] = np.sum(arr)
     
     EW = np.loadtxt("./data/ew_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc[i])),Rcc[j]))
      
     #################################
     axs1[i,j].scatter(r_los, EW, color='darkgray', facecolors='none', s=50) 
     
     axs1[i,j].plot(r_perp, nielsen(r_perp), lw=lw, color='blue', zorder=100, label='Nielsen')
     axs1[i,j].plot(r_perp, nielsen_l(r_perp), lw=lw, color='blue', ls=':', zorder=100) 
     axs1[i,j].plot(r_perp, nielsen_u(r_perp), lw=lw, color='blue', ls=':', zorder=100) 

     axs1[i,j].plot(r_perp, huang(r_perp), lw=lw, color='green', zorder=100, label='Huang')  
     axs1[i,j].plot(r_perp, huang_l(r_perp), lw=lw, color='green', ls=':', zorder=100)
     axs1[i,j].plot(r_perp, huang_u(r_perp), lw=lw, color='green', ls=':', zorder=100)
       
     axs1[i,j].plot(r_perp, dutta(r_perp), lw=lw, color='magenta', zorder=100, label='Dutta') 
     axs1[i,j].plot(r_perp, dutta_l(r_perp), lw=lw, color='magenta', ls=':', zorder=100)
     axs1[i,j].plot(r_perp, dutta_u(r_perp), lw=lw, color='magenta', ls=':',zorder=100)
     
     ####################################
     axs2[i,j].scatter(r_los, col_den, color='darkgray', facecolors='none', s=50)
     
     z,D,EW,errEW,logN,errN = np.loadtxt("coshalo_MgII.txt", skiprows=1, unpack=True)
     condition = np.logical_and(errN <1, errN >-1)
    
     axs2[i,j].scatter(D[condition], 10**logN[condition], s=50, color=color[4], label='COS-Halos', zorder=100)
     axs2[i,j].errorbar(D[condition], 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[4], elinewidth=2, zorder=100)

     condition = errN ==-1
     axs2[i,j].scatter(D[condition], 10**logN[condition], marker='^', s=50, color=color[4], zorder=100)

     condition = errN ==1
     axs2[i,j].scatter(D[condition], 10**logN[condition], marker='v', s=50, color=color[4], zorder=100)

     D,logN,errN = np.loadtxt("Paper4Table1.txt", skiprows=2, usecols=(5,15,16), unpack=True)
     #condition = np.logical_and(errN > 0.0, errN<1.)
     condition = errN > 0.0
     axs2[i,j].scatter(D[condition]*rCGM, 10**logN[condition], s=50, color=color[6], label='MAGIICAT', zorder=100)
     axs2[i,j].errorbar(D[condition]*rCGM, 10**logN[condition], yerr=10**(logN[condition]+ np.log10(1.-10**(-errN[condition]))), linestyle='None', ecolor=color[6], elinewidth=2, zorder=100)
     
     logN,errl,erru,D = np.loadtxt("CUBSVI_MgII.txt", skiprows=1, usecols=(1,2,3,4), unpack=True)
     axs2[i,j].scatter(D*rCGM, 10**logN, s=50, color=color[9], label='CUBS:VI', zorder=100)
     axs2[i,j].errorbar(D*rCGM, 10**logN, yerr=np.array([10**(logN+np.log10(1.-10**(-errl))),10**(logN+np.log10(1.-10**(-erru)))]), linestyle='None', ecolor=color[9], elinewidth=2, zorder=100)
    
     ###############################
     
     axs1[i,j].set_yscale('log')
     axs2[i,j].set_yscale('log')
     axs1[i,j].set_xscale('log')
     axs2[i,j].set_xscale('log')
     axs1[i,j].set_xlim(10,300)
     axs2[i,j].set_xlim(10,300)
     axs1[i,j].set_ylim(1.e-2, 2.e1)
     axs2[i,j].set_ylim(1.e11, 3.e15)
     if j !=0 :
        axs1[i,j].set_yticklabels(labels=[], minor=False)
        axs2[i,j].set_yticklabels(labels=[], minor=False)
        
     axs1[i,j].tick_params(which='major', axis='both', direction='inout', length=8, width=2, labelsize=12)
     axs1[i,j].tick_params(which='minor', axis='both', direction='inout', length=4, width=1, labelsize=12)
     axs2[i,j].tick_params(which='major', axis='both', direction='inout', length=8, width=2, labelsize=12)
     axs2[i,j].tick_params(which='minor', axis='both', direction='inout', length=4, width=1, labelsize=12)
     axs1[i,j].grid(axis='both', which='major')
     axs2[i,j].grid(axis='both', which='major')
     
axs1[2,1].set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=16)
axs2[2,1].set_xlabel(r'Impact parameter ($R_{\perp}$) [kpc]', size=16)
axs1[1,0].set_ylabel(r'EW MgII $[\rm \AA]$', size=16)
axs2[1,0].set_ylabel(r'$\rm N_{MgII} \, [\rm cm^{-2}]$', size=16)
axs1[0,2].legend(loc='upper right', ncol=3, fontsize=13)
axs2[0,2].legend(loc='upper left', ncol=1, fontsize=13)

fontsize = 20

fig1.text(0.06,0.975,r"$R_{\rm cc} \, \rm (kpc) \rightarrow$", fontsize=fontsize, color=colors)         
fig1.text(0.2,0.975,"{}".format(Rcc[0]),fontsize=fontsize, color=colors) 
fig1.text(0.51,0.975,"{}".format(Rcc[1]),fontsize=fontsize, color=colors) 
fig1.text(0.82,0.975,"{}".format(Rcc[2]),fontsize=fontsize, color=colors)  
fig1.text(0.015,0.97,r"$N_{\rm cc}$" ,fontsize=fontsize, color=colors) 
fig1.text(0.015,0.94,r"$\downarrow$" ,fontsize=fontsize, color=colors)  
fig1.text(0.015,0.80,r"$10^2$",fontsize=fontsize, color=colors)
fig1.text(0.015,0.5,r"$10^3$",fontsize=fontsize, color=colors)
fig1.text(0.015,0.2,r"$10^4$",fontsize=fontsize, color=colors)  

fig1.subplots_adjust(left= 0.06, bottom = 0.07, right=0.995,top=0.97,wspace=0,hspace=0)

fig2.text(0.06,0.975,r"$R_{\rm cc} \, \rm (kpc) \rightarrow$", fontsize=fontsize, color=colors)         
fig2.text(0.2,0.975,"{}".format(Rcc[0]),fontsize=fontsize, color=colors) 
fig2.text(0.51,0.975,"{}".format(Rcc[1]),fontsize=fontsize, color=colors) 
fig2.text(0.82,0.975,"{}".format(Rcc[2]),fontsize=fontsize, color=colors)  
fig2.text(0.015,0.97,r"$N_{\rm cc}$" ,fontsize=fontsize, color=colors) 
fig2.text(0.015,0.945,r"$\downarrow$" ,fontsize=fontsize, color=colors)  
fig2.text(0.015,0.82,r"$10^2$",fontsize=fontsize, color=colors)
fig2.text(0.015,0.53,r"$10^3$",fontsize=fontsize, color=colors)
fig2.text(0.015,0.23,r"$10^4$",fontsize=fontsize, color=colors)  

fig2.subplots_adjust(left= 0.06, bottom = 0.07, right=0.995,top=0.97,wspace=0,hspace=0)

fig1.savefig('./figures/EW_all_p_M{}.pdf'.format(int(np.log10(Mcold))))
#fig1.savefig('./figures/EW_all_p_M{}.png'.format(int(np.log10(Mcold))))
fig2.savefig('./figures/Col_all_p_M{}.pdf'.format(int(np.log10(Mcold))))
#fig2.savefig('./figures/Col_all_p_M{}.png'.format(int(np.log10(Mcold))))
plt.show()
plt.close()


# Fig 2 in paper

import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from decimal import Decimal

data_gen = False

f_V = np.array([0.1,0.01,0.001]) #volume fraction of clouds
N_cl = np.array([1e2,1e4,1e6]) #total number of clouds; fiducial value
#generate random number as coordinates of cubical clouds spread uniformly in space

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

if data_gen :
  file = h5py.File("data_cube.hdf5", "a") 
  for i in range(f_V.shape[0]):
    for j in range(N_cl.shape[0]):
        N = round((N_cl[j]/f_V[i])**(1./3.)) #total number of vacancies in 1D

        X_coord = np.random.randint(0, high=N, size=int(N_cl[j]), dtype=int)
        Y_coord = np.random.randint(0, high=N, size=int(N_cl[j]), dtype=int)
        Z_coord = np.random.randint(0, high=N, size=int(N_cl[j]), dtype=int)

        lbyL =  (f_V[i]/N_cl[j])**(1./3)

        cnt = np.zeros((N,N), dtype=int)   #for counting no of clouds along los

        for n in range(int(N_cl[j])):
            cnt[X_coord[n],Y_coord[n]] += 1      
       
        file.create_dataset("X_-{}_{}".format(round(np.log10(1/f_V[i])), round(np.log10(N_cl[j]))), data = X_coord) 
        file.create_dataset("Y_-{}_{}".format(round(np.log10(1/f_V[i])), round(np.log10(N_cl[j]))), data = Y_coord)
        file.create_dataset("Z_-{}_{}".format(round(np.log10(1/f_V[i])), round(np.log10(N_cl[j]))), data = Z_coord)
        file.create_dataset("cnt_-{}_{}".format(round(np.log10(1/f_V[i])), round(np.log10(N_cl[j]))), data = cnt)
    
        print("complete!\n")    
  file.close()

fig, axs = plt.subplots(4,3, figsize=(15,20))
color = ["blue", "magenta", "green"]

for i in range(f_V.shape[0]):
    for j in range(N_cl.shape[0]):
        N = round((N_cl[j]/f_V[i])**(1./3.)) #total number of vacancies in 1D

        X_coord = np.random.randint(0, high=N, size=int(N_cl[j]), dtype=int)
        Y_coord = np.random.randint(0, high=N, size=int(N_cl[j]), dtype=int)
        Z_coord = np.random.randint(0, high=N, size=int(N_cl[j]), dtype=int)

        lbyL =  (f_V[i]/N_cl[j])**(1./3)

        cnt = np.zeros((N,N), dtype=int)   #for counting no of clouds along los

        for n in range(int(N_cl[j])):
            cnt[X_coord[n],Y_coord[n]] += 1
            patch = plt.Rectangle([X_coord[n]*lbyL, Y_coord[n]*lbyL], lbyL, lbyL, color=color[j],linestyle='none',alpha=0.3)
            axs[j,i].add_patch(patch)
        axs[j,i].set_xticks([])
        axs[j,i].set_yticks([])
        
        expect = f_V[i]**(2./3)*N_cl[j]**(1./3.)
        mean = np.mean(cnt.flatten())
        var = np.var(cnt.flatten())
        mam = max(cnt.flatten())
        mim = min(cnt.flatten())
        print("mean :", mean)
        print("variance :", var)
        print("expectation :", expect)
        print("min",mim)
        print("max", mam)
      
        x = np.arange(mim,mam+1,1)
        
        pois = np.array([expect**x[k] * np.exp(-expect) / math.factorial(x[k]) for k in range(x.shape[0])])
        axs[3,i].hist(cnt.flatten(), bins=np.arange(mim,mam+1,1,dtype=int), density=True, histtype='step', align='left', color=color[j], stacked=False, linewidth=5, log=True)
        axs[3,i].plot(x, pois, color=color[j], linestyle='--', linewidth=5, zorder=100)
        
        print("complete!\n")     
    
    axs[3,i].grid(which='major',axis='both') 
    axs[3,i].tick_params(axis='both', labelsize=20)
    if i !=0:
       axs[3,i].set_yticklabels([])
    axs[3,i].set_ylim(1e-3,1.2)   

axs[3,0].set_xticks(np.arange(0, 40, 5))
axs[3,1].set_xticks(np.arange(0, 13, 4))
axs[3,2].set_xticks(np.arange(0, 7, 2))
axs[3,0].set_xlim(-0.1,37)
axs[3,1].set_xlim(-0.1,13)
axs[3,2].set_xlim(-0.1,6)
axs[3,0].set_yticks([1e-3,1e-2,1e-1,1])
axs[3,0].tick_params(which='major',axis='both', length=4, width=1, labelsize=25)
axs[3,0].tick_params(which='minor',axis='both', length=2, width=1, labelsize=25)  
axs[3,1].set_xlabel('number of cloudlets along a LOS', fontsize=28) 
fig.text(0.06,0.975,r"$f_V \rightarrow$" ,fontsize=35)         
fig.text(0.2,0.975,r"$10^{%d}$"%fexp(f_V[0]),fontsize=35) 
fig.text(0.5,0.975,r"$10^{%d}$"%fexp(f_V[1]),fontsize=35) 
fig.text(0.79,0.975,r"$10^{%d}$"%fexp(f_V[2]),fontsize=35)  
fig.text(0.004,0.92,r"$N_{cl}$" ,fontsize=35) 
fig.text(0.004,0.89,r"$\downarrow$" ,fontsize=35)  
fig.text(0.001,0.80,r"$10^{%d}$"%fexp(N_cl[0]),fontsize=35)
fig.text(0.001,0.6,r"$10^{%d}$"%fexp(N_cl[1]),fontsize=35)
fig.text(0.001,0.37,r"$10^{%d}$"%fexp(N_cl[2]),fontsize=35)  

plt.subplots_adjust(left= 0.06, bottom = 0.04, right=0.995,top=0.97,wspace=0.01,hspace=0.01)
plt.subplots_adjust(wspace=0.01,hspace=0.01)
plt.savefig("cubical_clouds.png")
#plt.savefig("cubical_clouds.pdf")
#plt.savefig("cubical_clouds_high_res.png", dpi=600.)

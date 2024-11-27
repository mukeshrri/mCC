# Another variant of Fig 11 in paper

import numpy as np
import matplotlib.pyplot as plt
import pickle

mu = 0.6
mp = 1.6726e-24
Msun = 1.989e33
pc = 3.086e18
kpc = 1.e3*pc
kms = 1.e5
kB = 1.3807e-16
amu = 1.6737e-24
c = 2.9979e10

rCGM = 280
Mcc = 1.e7
n_cold = 0.01

Rcc = 15
rc = np.array([10,20,40,60])

fig, axs = plt.subplots(figsize=(10,8))
color=['deepskyblue','olive','peru','orange','lightcoral','lawngreen','plum','tan','magenta']

meanturb = []
sigturb = []
no_empty = []

for ii in range(rc.shape[0]):
   fl = open("./data/dataturb_{}_{}_{}_{}.pickle".format(n_cold,Rcc,rc[ii],np.log10(Mcc)),'rb')
   param = pickle.load(fl)
   fl.close()
   cnt = 0
   b_turb = []
   for i in range(len(param)):
      data = param[i]
  
      if isinstance(data, np.ndarray):
    
        tot_lines = int(data.shape[0]/3.)
        for j in range(tot_lines):
            b_turb.append(data[3*j+1])
      
      else :
          cnt += 1    
   
   no_empty.append(cnt)
   #print("no of empty sightlines for {} kpc is {}".format(Rcc[ii],cnt)) 
   #print("area covering fraction for {} kpc is {}".format(Rcc[ii],(1.e4-cnt)/1.e4))    
   S_turb = np.array(b_turb)/np.sqrt(2.)
   condition = S_turb>0
   s_turb = S_turb[condition]
   meanturb.append(np.mean(s_turb))
   sigturb.append(np.std(s_turb))
   
   ratio = np.mean(s_turb) / np.std(s_turb)
   
   print('ratio is :', ratio)
 
   plt.hist(s_turb, bins=np.arange(0,max(s_turb),1), align='mid', color=color[ii], density=True, stacked=True, histtype='step', lw=4, ls='-', label=r'$r_{{\rm c}}={}$'.format(rc[ii]))

axs.set_xlim(.5,6)
axs.set_xscale('log')    
axs.set_yscale('log')
axs.grid()
axs.tick_params('both', length=6, width=2, which='major', labelsize=15)
axs.tick_params('both', length=4, width=1, which='minor', labelsize=15) 
axs.set_xlabel(r'$\sigma_{\rm turb} \, \rm (km \, s^{-1}$)', fontsize=15)
axs.set_ylabel(r'Normalized frequency', fontsize=15)
axs.legend(loc='upper right', fontsize=15)
fig.tight_layout()
plt.savefig('./figures/sigma_turb_hist1.pdf')
plt.show()
plt.close()   


Nc = Mcc*Msun/(4./3.*np.pi*rc**3.*pc**3.*n_cold*mu*mp)
fa = Nc*(rc*1.e-3/Rcc)**2. 

meanturb = np.array(meanturb)
sigturb = np.array(sigturb)
siggauss = np.array([0.45,0.6,0.8,1.3])

fig, axs = plt.subplots(figsize=(8,6))

fa_T = (1.e4-np.array(no_empty))/1.e4
print(fa_T)

for i in range(rc.shape[0]):
   axs.scatter(fa_T[i], meanturb[i]/sigturb[i], s=300, color=color[i], edgecolors='black', label=r'$r_{{\rm cl}}={}$'.format(rc[i]))
   #axs.scatter(rc[i], meanturb[i]/sigturb[i], s=300, color=color[i], edgecolors='black', label=r'$r_{{\rm c}}={}$'.format(rc[i]))
  
axs.set_ylabel(r'$\frac{\mathrm{mean}(\sigma_{\rm turb})}{\mathrm{std}(\sigma_{\rm turb})}$', fontsize=24)

axs.hlines(1., 0, 1.1, color='black', ls='--', lw=3)

axs.set_xlabel(r'Area covering fraction ($f_A^{\rm cl}$)', fontsize=18)

axs.set_ylim(0.3,1.6)
axs.grid()
axs.tick_params('both', length=8, width=3, which='major', labelsize=18)
axs.tick_params('both', length=4, width=2, which='minor', labelsize=18) 
axs.legend(ncol=2, loc='upper left', fontsize=18)
fig.tight_layout()
#plt.savefig('./figures/fac_sigratio1.png')
plt.savefig('figures/fac_sigratio1.pdf')
plt.show()
plt.close()

fig, axs = plt.subplots(figsize=(10,8))

#axs.scatter(Rcc, fa*0.1, color='red', s=200, label=r'$f_A \times 0.1$')
#axs.scatter(Rcc[-1], fa[-1]*0.1, color='red', edgecolors='black', s=200)

#axs.scatter(Rcc, siggauss, color='orange', s=200, label=r'$\sigma_{\rm gauss}$')
#axs.scatter(Rcc[-1], siggauss[-1], color='orange', edgecolors='black', s=200)

axs.scatter(rc, meanturb, color='magenta', s=200, label=r'mean$(\sigma_{\rm turb})$')
#axs.scatter(Rcc[-1], meanturb[-1], color='magenta', edgecolors='black', s=200)

axs.scatter(rc, sigturb, color='deepskyblue', s=200, label=r'std$(\sigma_{\rm turb})$')
#axs.scatter(Rcc[-1], sigturb[-1], color='skyblue', edgecolors='black', s=200)

axs.set_title(r'R$_{{\rm cc}}$ = {} kpc'.format(Rcc), fontsize=24)
#axs.set_xlim(1.e11,5.e13)
#axs.set_ylim(2.e-3,0.5)
#axs.set_xscale('log')    
#axs.set_yscale('log')
axs.grid()
axs.tick_params('both', length=6, width=2, which='major', labelsize=20)
axs.tick_params('both', length=4, width=1, which='minor', labelsize=20) 
axs.set_xlabel(r'$r_{\rm cl}$ (pc)', fontsize=24)
axs.legend(loc='upper right', fontsize=16)
fig.tight_layout()
#plt.savefig('figures/sigmaturb1.png')
plt.show()
plt.close()   


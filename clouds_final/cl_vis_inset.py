# Fig 13 in paper

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import h5py
import math
from decimal import Decimal
from scipy.spatial.transform import Rotation as rot
from scipy.spatial.distance import squareform,pdist
import multiprocessing
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()
    
mu = 0.67
mp = 1.67e-24
Msun = 1.989e33
kpc = 3.086e21
fac = mu*mp*kpc**3/Msun
proc = 7

N_los = 10000
N_cc = 1000

M_cold = 1.e10
n_cold = 0.01
rCGM = 280

D_min = 10
D_max = 280
alpha = 1.2

d_min = 0.1
d_max = 10
beta = 0.2

a_min = 0.01
a_max = 1
gamma = 0.2

def cloud_plot(data_):
    x0=data_[0]
    y0=data_[1]
    a=data_[2]
    b=data_[3]
    c=data_[4]
    angl_alfa=data_[5]
    angl_beta=data_[6]
    angl_gmma=data_[7]   
    X_plot = []
    Y_plot = []  
    RM = np.zeros((3,3))
    RM[0,0] = np.cos(angl_beta)*np.cos(angl_gmma)
    RM[1,0] = np.sin(angl_alfa)*np.sin(angl_beta)*np.cos(angl_gmma)- np.cos(angl_alfa)*np.sin(angl_gmma)
    RM[2,0] = np.cos(angl_alfa)*np.sin(angl_beta)*np.cos(angl_gmma)+ np.sin(angl_alfa)*np.sin(angl_gmma)
    RM[0,1] = np.cos(angl_beta)*np.sin(angl_gmma)
    RM[1,1] = np.sin(angl_alfa)*np.sin(angl_beta)*np.sin(angl_gmma)+ np.cos(angl_alfa)*np.cos(angl_gmma)
    RM[2,1] = np.cos(angl_alfa)*np.sin(angl_beta)*np.sin(angl_gmma)- np.sin(angl_alfa)*np.cos(angl_gmma)
    RM[0,2] = -np.sin(angl_beta)
    RM[1,2] = np.sin(angl_alfa)*np.cos(angl_beta)
    RM[2,2] = np.cos(angl_alfa)*np.cos(angl_beta)
    
    mam = a_max
    
    X_LOS = np.linspace(x0-2*mam,x0+2*mam,int(100*mam))
    Y_LOS = np.linspace(y0-2*mam,y0+2*mam,int(100*mam))
    
    for i in range(X_LOS.shape[0]):
       for j in range(Y_LOS.shape[0]):
           X = RM[0,0]*(X_LOS[i]-x0) + RM[0,1]*(Y_LOS[j]-y0) 
           Y = RM[1,0]*(X_LOS[i]-x0) + RM[1,1]*(Y_LOS[j]-y0) 
           Z = RM[2,0]*(X_LOS[i]-x0) + RM[2,1]*(Y_LOS[j]-y0)
        
           a_ = (RM[0,2]/a)**2. + (RM[1,2]/b)**2. + (RM[2,2]/c)**2.
           b_ = 2.*(RM[0,2]*X/a**2. + RM[1,2]*Y/b**2. + RM[2,2]*Z/c**2.)
           c_ = (X/a)**2. + (Y/b)**2. + (Z/c)**2. - 1. 
        
           quant = b_**2. - 4.*a_*c_
           if quant>=0:
              X_plot.append(X_LOS[i])
              Y_plot.append(Y_LOS[j])
              
    return np.array((X_plot,Y_plot))          
  
with h5py.File("./data/data_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.hdf5".format(N_cc,int(np.log10(M_cold)),n_cold,rCGM,D_min,D_max,alpha,d_min,d_max,beta,a_min,a_max,gamma), 'r') as f:
     #X0 = np.array(f['X0'])
     #Y0 = np.array(f['Y0'])
     #Z0 = np.array(f['Z0'])
     x0 = np.array(f['x0'])
     y0 = np.array(f['y0'])
     #z0 = np.array(f['z0'])
     A = np.array(f['a'])
     B = np.array(f['b'])
     C = np.array(f['c'])
     angl_alfa = np.array(f['angl_alfa'])
     angl_beta = np.array(f['angl_beta'])
     angl_gmma = np.array(f['angl_gmma'])
     info = np.array(f['info'])
 
N_cl = info[5]  
print('no fo clouds:', N_cl)
data_cl = np.column_stack((x0, y0, A, B, C, angl_alfa, angl_beta, angl_gmma))
with multiprocessing.Pool(processes=proc) as pool:
     result = pool.map(cloud_plot,data_cl)           

fig, axs = plt.subplots(figsize=(10,10))

ax_new = fig.add_axes([0.6, 0.6, 0.38, 0.38])

for i in range(int(N_cl)):
   XY = result[i]
   X = XY[0,:]
   Y = XY[1,:]
   axs.plot(X,Y, color='blue', linestyle='None', marker='.', markersize=1, alpha=0.2)
   ax_new.plot(X,Y, color='blue', linestyle='None', marker='.', markersize=1, alpha=0.5, zorder=100)

ax_new.set(title='', xlim=(97,118), ylim=(223,241), xticks=[], yticks=[])
           
patch = plt.Circle([0,0], rCGM, color='black', fill=False, alpha=1.0) 
axs.add_patch(patch) 

axs.set_xlim(-rCGM,rCGM)
axs.set_ylim(-rCGM,rCGM) 
axs.set_ylabel('Y [kpc]', fontsize=18)
axs.set_xlabel('X [kpc]', fontsize=18) 
axs.tick_params('both', length=6, width=2, which='major', labelsize=16)
axs.tick_params('both', length=4, width=1, which='minor', labelsize=16) 
fig.tight_layout()  
plt.savefig('./figures/cloud_vis_l.png')
plt.savefig('./figures/cloud_vis_l.pdf')
#plt.show()
plt.close()  
print("--- %s seconds ---" % (time.time() - start_time))

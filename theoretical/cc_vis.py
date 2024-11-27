# Visualising the cloud complexes in CGM (Fig 1 in paper)

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

rCGM = 280
Rcc = 10
Ncc = 1.e3    
Mcold = 1.e10      

x,y,z = np.loadtxt("./data/coord_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc),unpack=True)

fig, axs = plt.subplots(figsize=(10,10))

for i in range(int(Ncc)):
   patch = plt.Circle([x[i],y[i]], Rcc, color='deepskyblue', fill=True, alpha=0.5)
   axs.add_patch(patch) 
           
patch = plt.Circle([0,0], rCGM, color='black', fill=False, alpha=1.0) 
plt.text(-280,260,r'power-law,$\alpha=1.2$', fontsize=21, color='black') 
axs.add_patch(patch) 
axs.set_xlim(-rCGM,rCGM)
axs.set_ylim(-rCGM,rCGM) 
axs.set_ylabel('Y [kpc]', fontsize=20)
axs.set_xlabel('X [kpc]', fontsize=20) 
axs.tick_params('both', length=6, width=2, which='major', labelsize=20)
axs.tick_params('both', length=4, width=1, which='minor', labelsize=20) 
fig.tight_layout()  
plt.savefig('./figures/cc_p.pdf')
plt.show()
plt.close()  
print("--- %s seconds ---" % (time.time() - start_time))

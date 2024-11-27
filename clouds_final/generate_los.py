# generate LOSs

import os
import sys
import numpy as np
import h5py
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()

N_los = 1.e4
rCGM = 280

r_los = np.logspace(1,np.log10(rCGM),int(N_los))
th_los = np.random.uniform(low=0, high=2.*np.pi, size=int(N_los))
x_los = r_los*np.cos(th_los)
y_los = r_los*np.sin(th_los) 
         
fl = h5py.File("./data/los_data_4.hdf5", "a")
fl.create_dataset('x_los', data = x_los)
fl.create_dataset('y_los', data = y_los)
fl.close()  

print('complete! \n')
print("--- %s seconds ---\n " % (time.time() - start_time))      

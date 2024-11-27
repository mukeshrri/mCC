# Solid black line data in top right panel of Fig 14 in paper

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

Mcold = 1.e10
Ncc = 1.e3
Rcc = 10
Mcc = Mcold/Ncc

EW = np.loadtxt("./data/ew_pl_{}_{}_{}.txt".format(int(np.log10(Mcold)),int(np.log10(Ncc)),Rcc))

r_los = np.logspace(np.log10(10),np.log10(rCGM),int(Nlos))

impact = np.arange(0,220,20)
cov_frac_b = np.zeros(impact.shape[0]-1)
impact_b = np.zeros(impact.shape[0]-1)
for i in range(impact.shape[0]-1):
      condition1 = np.logical_and(r_los > impact[i], r_los <= impact[i+1])
      condition = np.logical_and(condition1, EW > 0.3)
      cov_frac_b[i] = EW[condition].shape[0]/r_los[condition1].shape[0]
      impact_b[i] = (impact[i] + impact[i+1])/2.
      
np.savetxt('cov_frac_mCC.txt', np.column_stack((impact_b,cov_frac_b)))      
      

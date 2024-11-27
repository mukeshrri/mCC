# Generate absorption profile and EW

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import multiprocessing
import warnings
import time
from itertools import product
import pickle
from scipy.special import wofz

kms = 1.e5
c = 2.9979e10
me = 9.1094e-28
e = 4.8032e-10
mp = 1.6726e-24
amu = 1.67377e-24
kB = 1.3807e-16

def generate_norm_flux(optical_depth):
    norm_flux = np.exp(-1.0 * optical_depth)
    return norm_flux

def return_EW(wavel, norm_flux):
    EW = np.trapz((1. - norm_flux), wavel) # in A
    return EW

def return_voigt(x,y):
    z = x + 1j * y
    profile = wofz(z).real
    return profile
    
def return_optical_depth(
    f_lu, # oscillator strength
    N_l, # column density
    gamma_ul, # probability coefficients
    b, # doppler width in km/s
    wave0, # rest-frame transition in Angstrom
    v_los, # line of sight velocity offset
    spectral_res, # spectral resolution in km/s
    ):
    
    vel_max = np.abs(v_los) + 10.*b
    nu0 = c / (wave0*1.e-8)
    nu0_ob = nu0 * (1. - v_los*kms/c)
    wave0_ob = wave0 * (1. + v_los*kms/c)
    
    pre_factor = np.sqrt(np.pi) * e**2 / (me*c) 
    pre_factor = pre_factor * f_lu * (N_l * wave0_ob*1.e-8) / (b*kms)
    
    vel_range = np.arange(-vel_max, vel_max+spectral_res, spectral_res)
    wave_range = wave0_ob * (1. + vel_range*kms/c)
    nu = c / (wave_range * 1.e-8)
    
    x = (nu/nu0_ob - 1.)*c/(b*kms)
    y = gamma_ul/(4.*np.pi*nu0_ob) * (c/(b*kms))
    voigt_profile = return_voigt(x,y)
    
    optical_depth = pre_factor * voigt_profile
    
    return (nu, optical_depth)

def return_coarse_flux(nu, norm_flux, spectral_res, wave0):
    wave = c / nu * 1.e8
    wave_res = wave0 * spectral_res*kms/c
    nbins = int((np.max(wave) - np.min(wave)) // wave_res)
    wave_coarse = np.linspace(np.min(wave), np.max(wave), nbins+1)
    flux_coarse = np.zeros(nbins)
    for i in range(nbins):
        if i<nbins-1:
            condition = np.logical_and(wave>=wave_coarse[i], wave<wave_coarse[i+1])
        else :    
            condition = np.logical_and(wave>=wave_coarse[i], wave<=wave_coarse[i+1])
           
        bin_width = np.max(wave[condition]) - np.min(wave[condition])  
          
        flux_coarse[i] = np.trapz(norm_flux[condition], wave[condition])/bin_width
    wave_coarse = 0.5*(wave_coarse[1:] + wave_coarse[:-1])
    half_width = [0.5*(wave_coarse[1]-wave_coarse[0]), 0.5*(wave_coarse[-1]-wave_coarse[-2])]
    wave_coarse = np.hstack((wave_coarse[0]-half_width[0], wave_coarse, wave_coarse[-1]+half_width[1]))
    flux_coarse = np.hstack((flux_coarse[0], flux_coarse, flux_coarse[-1]))
    return (wave_coarse, flux_coarse)    
    

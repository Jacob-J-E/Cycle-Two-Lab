#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:49:01 2021

@author: dawudabd-alghani
"""

import os
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

def intensity(filename, wavelength):
    conc = np.loadtxt(filename, skiprows=33, delimiter=',', max_rows=3648)
    intensity = 0
    i = 0
    j = 0
    while conc[i,0] < wavelength+0.5:
        i += 1
        while conc[i,0] < wavelength-0.5:
            intensity += conc[i,1]
            i += 1
            j +=1
    return(intensity/j)
    # while int(conc[i,0]) != wavelength:
    #     i += 1
    # return(conc[i,1])
def intensities(wavelength):
    intensities = np.array([])
    absolute_path = os.path.abspath(os.path.dirname('White LED.py'))
    for filename in os.listdir(absolute_path+"/White_LED_Spectra"):
        if filename[-3] == 'c':
            filename = 'White_LED_Spectra/' + filename
            intensities = np.append(intensities, abs(intensity(filename, wavelength)))
    return(intensities/max(intensities))
def exponential(x,a):
    return np.exp(a*x)
def epsilon(wavelength):
    concentrations = np.array([8.097,17.658,3.514,4.67,2.667,10.998,9.401,1.098,14.909,20.865,6.050,2.035,0,13.117])
    intensities2 = intensities(wavelength)
    sigma = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    p0 = [-1]
    params, cov = sp.curve_fit(exponential, concentrations, intensities2, p0, sigma=sigma)
    # plt.grid()
    # plt.errorbar(concentrations, intensities2, xerr = 0, yerr = 0.005, fmt = 'x')
    # plt.xlabel('Concentration (g/L)')
    # plt.ylabel('Normalised Intensity')
    # plt.plot(np.sort(concentrations), exponential(np.sort(concentrations), *params))
    return(params/-0.054)

epsilon_values = np.array([])
wavelengths = np.arange(400,700)
for i in range(400,700):
    epsilon_values = np.append(epsilon_values, epsilon(i))

plt.grid()
plt.plot(wavelengths, epsilon_values, '.')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Molar extinction coefficient (m\u00b2/mol)')
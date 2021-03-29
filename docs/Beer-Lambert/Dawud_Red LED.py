#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:59:32 2021

@author: dawudabd-alghani
"""
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import os

def exponential(x,a,b):
    return a*np.exp(b*x)

def red_LED():
    red_LED_conc = np.array([0,1.059,1.557,2.152,2.665,3.234,3.877,4.358,5.110,5.538,
                             6.237,6.651,7.157,8.008,9.158,10.246,11.348,13.408,15.447,
                             16.974,19.622,21.891,24.310,28.527,33.334])
    red_LED_peak = np.array([0.72,0.63,0.68,0.63,0.59,0.56,0.53,0.54,0.50,0.46,0.44,
                             0.42,0.40,0.37,0.34,0.32,0.29,0.25,0.20,0.19,0.16,0.15,
                             0.13,0.11,0.09])/0.72
    p0 = [1,-1]
    sigma = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
             0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    params, cov = sp.curve_fit(exponential, red_LED_conc, red_LED_peak, p0, sigma=sigma)
    plt.errorbar(red_LED_conc, red_LED_peak, xerr = 0, yerr = 0.005, fmt = 'x')
    plt.plot(red_LED_conc, exponential(red_LED_conc, *params))
    plt.xlabel('Concentration (g/L)')
    plt.ylabel('Normalised peak intensity')
    print(params[1]/-0.054,'±', np.sqrt(cov[1,1])/0.054)

red_LED()
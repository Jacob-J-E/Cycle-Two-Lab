#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 19:19:20 2021

@author: dawudabd-alghani
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

u = np.array([233,176,372,136,127,473,123,584,119,682,154,302])
v = np.array([172,224,129,365,474,128,578,117,678,115,294,146])
sigma = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
x = 1/u
y = 1/v
plt.plot(x,y,'x')
plt.grid()
plt.xlabel('1/s (mm)')
plt.ylabel('1/s\' (mm)')
inital = [-1,0]
def linear(x,m,c):
    return m*x+c
p0,cov = sp.curve_fit(linear,x,y,inital,sigma=sigma)
plt.plot(x,linear(x,*p0))
print(p0[0],'±',np.sqrt(cov[0,0]))
print(p0[1],'±',np.sqrt(cov[1,1]))
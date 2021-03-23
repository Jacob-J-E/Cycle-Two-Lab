#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:12:51 2021

@author: dawudabd-alghani
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

m = np.arange(-5,7)
mins = np.array([-6.73,-5.59,-4.25,-3.13,-1.88,-0.56,0.69,1.97,3.49,4.49,5.41,6.56])/10**3
unc = mins*0.01

fit,cov = np.polyfit(m, mins, 1, w=1/unc, cov=1)
linear = np.poly1d(fit)

plt.grid()
plt.xlabel('m value')
plt.ylabel('Shifted minimum location (mm)')
plt.plot(m, mins, 'x')
plt.plot(m, linear(m))
print(linear[1],'±', np.sqrt(cov[1,1]))

λ = 670/10**9
unc_λ = 1/10**9
d = 230/10**6
unc_d = 20/10**6
f = 500/10**3
unc_f = 1/10**3

grad = λ*f/d
unc = grad * np.sqrt((unc_λ/λ)**2 + (unc_d/d)**2 + (unc_f/f)**2)
print(grad, '±', unc)


# data = np.loadtxt('Multi Slit/second_Values.csv', skiprows=1, delimiter=',')
# x = data[:,0]
# y = data[:,1]

# plt.plot(x,y,'.')

# def sinc(x,a,b,c,d,e):
#     x = x - e
#     return a * (np.sin(b*x)/(b*x))**2 * (np.cos(c*x))**2 + d

# p0 = [130,0.008,0.039,0.01,515]
# params, cov = sp.curve_fit(sinc, x, y, p0)
# plt.plot(x, sinc(x, *params))
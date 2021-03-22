import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image
from scipy.optimize import curve_fit
from skimage.io import imread, imshow
from skimage import data

#Define Functions
def sinc(x):
    if x.any() == 0:
        return 1
    else:
        return np.sin(x)/x

def intensity_pattern(x,C,a,b):
    d = 0
    return 4*C*(sinc(a*(x-b))*np.cos(7619.534421*(x-b)))**2+d
#2156.9

x,y=np.loadtxt('PlotValues.csv',delimiter=',',unpack=1,skiprows=1)
z = x*5.2e-06

plt.scatter(z,y,marker='x',s=11)
C_guess = 1
sinc_coeff = 0.0023
#d_guess = 0
b_guess = 0.00185
# C_guess = 30
# sinc_coeff = 0.008
# b_guess = 0.0025
# d_guess = 0
fit_guess = [C_guess,sinc_coeff,b_guess]
po,po_cov=sp.optimize.curve_fit(intensity_pattern,z,y,fit_guess)
print(po)
plt.plot(z,intensity_pattern(z,po[0],po[1],po[2]),color='red')
plt.show()

L = z[70:630]
K = y[70:630]
plt.scatter(L,K)
C_guess2 = 1
sinc_coeff2 = 0.0023
b_guess2 = 0.00185
fit_guess2 = [C_guess2,sinc_coeff2,b_guess2]
po2,po_cov2=sp.optimize.curve_fit(intensity_pattern,L,K,fit_guess2)
plt.plot(L,intensity_pattern(L,po2[0],po2[1],po2[2]),color='red')
print(len(L))
print(len(K))
plt.show()

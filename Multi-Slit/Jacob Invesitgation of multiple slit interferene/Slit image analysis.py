#Written by Jacob J. Edginton
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

def intensity_pattern(x,C,a,e):
    d = -0
    #7619.534421
    return C*(sinc(a*x)*np.cos(e*x))**2+d
#2156.9

#load data
x,y=np.loadtxt(r'C:\Users\Ellre\Desktop\Physics Degree\Year 1\Lab\Term 2\Multi-Slit\AAA.csv',delimiter=',',unpack=1,skiprows=1)

#scale to meters and center at 0
z = x*5.2e-06-0.00183

#slice data so that it only is for the higher intensity region
L = z[70:630]
K = y[70:630]

#define initial guess
C_guess2 = 1
sinc_coeff2 = 0.2
e_guess2 = 7619
fit_guess2 = [C_guess2,sinc_coeff2,e_guess2]

#Fit curve
po2,po_cov2=sp.optimize.curve_fit(intensity_pattern,L,K,fit_guess2)
print(po2)

#Plot Results
plt.scatter(L,K,marker='x',s=20,label='Experimental Data')
plt.plot(L,intensity_pattern(L,po2[0],po2[1],po2[2]),color='red'label='Intensity Curve Fit'))
plt.xlabel('Position (m)')
plt.ylabel('Intensity (dimensionless)')
plt.grid()
plt.plot(L,130*np.cos(1000*L)**2,color='black',linestyle='--',label='Envelope')
plt.legend()
plt.show()

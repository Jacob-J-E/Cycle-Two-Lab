#Written by Jacob J. Edginton
#All package imports and function definitions are placed outside loops to redcuce the computational complexity of the code
import os
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def my_exp(t, exponent):
    return np.exp(-1*t*exponent*l)
    
def fit_func(x,a,mu,sig):
    gaus = a*np.exp(-(x-mu)**2/(2*sig**2))
    return gaus    

#Global variables
intensity_uncertainty = 0.01 
l = 5.4e-2 #length of box


#define the parameters of the code execution:

#guess for the exponential fitting
initial_guess = [0.07]

#starting wavelength for spectra
starting = 450

#number of wavelength integers to iterative through
N = 50


#Add concentration data and define empty arrays to be used in iteration
conc = np.array([0,1.098,10.988,13.117,14.909,17.658,2.035,2.667,20.865,3.514,4.670,6.050,8.097,9.401])
intensityArray = np.zeros((N,14),dtype=float)
norm = np.zeros_like(intensityArray)
b = intensity_uncertainty+np.zeros_like(norm[1])

#Main loop of the code
for j in range(0,N):
    fileCount = 0

    wavelength = starting+j
    absolute_path = os.path.abspath(os.path.dirname('White Light.py'))#finds the correct directory.
    
    for filename in os.listdir(absolute_path+"\\spectrumData"): #will loop through each file in folder
    
        fileCount += 1

        #load a CSV in each iteration
        data = np.loadtxt(open(absolute_path+ "\\spectrumData\\"+ filename,'rt').readlines()[:-1], skiprows=33, delimiter=',', unpack=True)
     
        for i in range(0, len(data[1])):
    
            if(int(data[0][i]) == wavelength):
                if data[1][i] < 0:
                    #if the condition is set to = 0 we seem to get a floating point error, this small number avoids that error
                   intensityArray[j][fileCount-1] = 0.000000001
                else:   
                   intensityArray[j][fileCount-1] = data[1][i]

              
                break
        #the intensity is normalized to 1 here
        norm[j] = intensityArray[j]/intensityArray[j][0]

        
            
#Create wavelength array
wav = np.arange(starting,starting+N,1)
extinct = []

#At each wavelength value find the extinction coefficent
for j in range(0,N):
    po,cov = curve_fit(my_exp, conc, norm[j],initial_guess)
    extinct.append(po[0])

extinct = np.array(extinct)


#Curve fit the spectra to a gaussian 
gauss_guess = [4,610,50]
bo,po_cov=sp.optimize.curve_fit(fit_func,wav[extinct>0],extinct[extinct>0],gauss_guess)
lin = np.arange(400,700,1)

#Plot results of curve fit
plt.scatter(wav[extinct>0],extinct[extinct>0],s = 5, marker = 'o',linewidth=1)
plt.plot(lin,fit_func(lin,bo[0],bo[1],bo[2]),color='blue')
plt.xlabel('Wavelegnth (nm)')
plt.ylabel(r'Extinction Coefficent $L(gm)^-1$')
plt.tight_layout()
plt.grid(linestyle = '-')
print('bo = ',bo,)
print('po_cov = ',po_cov,)
plt.show()






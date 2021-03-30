#Written by Jacob J. Edginton
#All package imports and function definitions are placed outside code blocks so that they are global
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#exponential curve 
def my_exp(t, exponent, factor):
    return factor*np.exp(t*exponent)

#Iterative uncertainty formula
def Uc1(M1,mc1,Ci1,UC1):
    uCiM = (mc1*(Ci1-1))/((1+Ci1)*(M1+mc1)**2)
    uCimc = M1*(1-Ci1)/((1+Ci1)*(M1+mc1)**2)
    uCi = (M1-mc1)/((M1+mc1)*(1+Ci1)**2)     
    d = np.sqrt((uCiM*sigma_M)**2+(uCimc*sigma_mc)**2+(uCi*UC1)**2)
    return d    
#%%
#This code block is for linear fitting
mono_conc,mono_peak = np.loadtxt('redled.txt',unpack=1)
norm_mono_ln = - np.log(mono_peak/mono_peak[0])
intensity_uncertainty = 0.01 
a = intensity_uncertainty+np.zeros_like(norm_mono_ln[1:])

#Find Linear Fit 
fit_mono,cov_mono = sp.polyfit(mono_conc[1:],norm_mono_ln[1:],1,w=1/a,cov=True)
sig_0_noisy1 = sp.sqrt(cov_mono[0,0]) #The uncertainty in the slope
sig_1_noisy1 = sp.sqrt(cov_mono[1,1]) #The uncertainty in the intercept
pnmono=sp.poly1d(fit_mono)

#plot results
plt.scatter(mono_conc[1:],norm_mono_ln[1:],marker='x')
plt.plot(mono_conc,pnmono(mono_conc))
plt.xlabel(r'Concentration ($gL^{-1}$)')
plt.ylabel('-ln(Intensity/Origional intensity) (dimensionless)')
plt.errorbar(mono_conc[1:],norm_mono_ln[1:],yerr = a,xerr=0, ls = 'none')
print('Linear fit: ',fit_mono,)
print('The uncertainty in the slope',sig_0_noisy1,)
print('The uncertainty in the intercept',sig_1_noisy1,)
#%%
#This code block is for uncertainties
mc = 0,0,0.53,0.25,0.3,0.26,0.29,0.33,0.25,0.39,0.23,0.37,0.23,0.28,0.46,0.62,0.6,0.62,1.13,1.15,0.92,1.54,1.4,1.54,2.58,3.05
M = 500.11,499.75,500.28,500.14,500.16,500.36,500.52,500.79,501,501.17,501.48,501.61,501.92,502.26,502.18,502.65,503.14,503.68,504.13,505.23,506.37,507.23,508.74,510.1,511.52,514.09
C = 0,0,1.059406732,1.556698974,2.151888397,2.664599999,3.233687034,3.877896423,4.358023694,5.110201093,5.538270883,6.236918022,6.650891898,7.157129216,8.008286223,9.158164713,10.24566477,11.34753672,13.40844874,15.44678001,16.9741503,19.6216564,21.89060991,24.31032815,28.52687065,33.33414419
C = np.array(C)
UC_0 = 2
sigma_M = 0.005
sigma_mc = 0.005
 
b = 0.1
uncert = []
for i in range(0,len(C)):
    b = Uc1(M[i]/1000,mc[i]/1000,C[i],b)
    uncert.append(b)

print('uncert = ',uncert,)   

p0=[-1, 1]
intensity_uncertainty = 0.01 
b = intensity_uncertainty+np.zeros_like(norm_mono_ln[1:])
#%%
#This code block is for exponential curve fitting

#fit curve
fit_exp = curve_fit(my_exp, mono_conc[1:], mono_peak[1:], p0=p0,sigma=b)
print(fit_exp)
data_fit_exp = my_exp(mono_conc, *fit_exp[0])

#plot fit results
plt.plot(mono_conc,data_fit_exp,color='green')
plt.scatter(mono_conc,mono_peak,marker='x',color='black')
plt.ylabel('Peak Intensity (dimensionless)')
plt.grid()
int_un = np.zeros_like(mono_peak)+0.009
plt.errorbar(mono_conc,mono_peak,yerr = int_un ,xerr=C*0.02, ls = 'none')
plt.show()
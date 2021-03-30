import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#load data and convert to natural log form
mono_conc,mono_peak = np.loadtxt('redled.txt',unpack=1)
norm_mono_ln = - np.log(mono_peak/mono_peak[0])
intensity_uncertainty = 0.01 



# Slice data such that only the linear range is plotted
mono_conc_slice_1 = mono_conc[1:]
norm_slice_1 = norm_mono_ln[1:]
plotx = mono_conc_slice_1[mono_conc_slice_1<20]
ploty = norm_slice_1[mono_conc_slice_1<20]

#define uncertainty
a = intensity_uncertainty+np.zeros_like(ploty)

#Use polyfit to find the linear fit of the data
fit_mono,cov_mono = np.polyfit(plotx,ploty,1,w=1/a,cov=True)
sig_0_noisy1 = np.sqrt(cov_mono[0,0]) #The uncertainty in the slope
sig_1_noisy1 = np.sqrt(cov_mono[1,1]) #The uncertainty in the intercept
pnmono=np.poly1d(fit_mono)

plt.scatter(plotx,ploty,marker='x',color='black')
plt.plot(plotx,pnmono(plotx))
plt.xlabel(r'Concentration ($gL^{-1}$)')
plt.ylabel('-ln(Intensity/Origional intensity) (dimensionless)')
plt.errorbar(plotx,ploty,yerr = a,xerr=0, ls = 'none')
plt.grid()

print('Fit parameters: ',fit_mono,)
print('Uncertainty in slope',sig_0_noisy1,)
print('Uncertainty in Intercept',sig_1_noisy1,)


#plt.show is written last as the rest of the code won't be executed unti the plot is closed
plt.show()
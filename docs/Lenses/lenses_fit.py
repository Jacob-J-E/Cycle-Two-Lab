#Written by Jacob J. Edginton
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

#define function
def linear(x,m,c):

    return m*x+c
#adds arrays of data 
u = np.array([233,176,372,136,127,473,123,584,119,682,154,302])
v = np.array([172,224,129,365,474,128,578,117,678,115,294,146])

#redefine as reciprocal
plotx = 1/u
ploty = 1/v


#plot scatter



#Set limits
plt.xlim(0.9*min(plotx),1.1*max(plotx))
plt.ylim(0.9*min(ploty),1.1*max(ploty))


#Set initial guess
inital = [-1,0]

#curve fit
po,cov = spo.curve_fit(linear,plotx,ploty,inital)
sig_0 = np.sqrt(cov[0,0]) #The uncertainty in the slope
sig_1 = np.sqrt(cov[1,1]) #The uncertainty in the intercept

#print fit parameters
print(po)
print(sig_0)
print(sig_1)

#Plot results
plt.plot(plotx,linear(plotx,po[0],po[1]))
plt.xlabel('1/s (1/mm)')
plt.ylabel("1/s' (1/mm)")
plt.scatter(plotx,ploty,color = 'red',marker='x')
plt.grid()
plt.show()

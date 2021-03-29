#imports numpy
import numpy as np
#imports matplotlib
import matplotlib.pyplot as plt
#imports scipy
import scipy.optimize as spo
#imports u array
u = np.array([233,176,372,136,127,473,123,584,119,682,154,302])
#imports v array
v = np.array([172,224,129,365,474,128,578,117,678,115,294,146])
#redefine in reciprocal
plotx = 1/u
#redefine in reciprocal
ploty = 1/v
#plot scatter
plt.scatter(plotx,ploty,color = 'red')
#plot grid
plt.grid()
#Set x limit
plt.xlim(0.9*min(plotx),1.1*max(plotx))
#Set y limit
plt.ylim(0.9*min(ploty),1.1*max(ploty))
#Set initial guess
inital = [-1,0]
#define function
def linear(x,m,c):
    #return function value
    return m*x+c
#curve fit
po,cov = spo.curve_fit(linear,plotx,ploty,inital)
#print fit parameters
print(po)
#plot curve fit
plt.plot(plotx,linear(plotx,po[0],po[1]))
#show plot
plt.show()
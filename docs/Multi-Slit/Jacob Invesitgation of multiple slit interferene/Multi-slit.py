#Written by Jacob J. Edginton
import numpy as np
import matplotlib.pyplot as plt

#Specify data
m = np.array([1,2,3,4,5,6,0,-1,-2,-3,-4,-5])
rel_spacing = np.array([0.69,1.97,3.49,4.49,5.41,6.56,-0.56,-1.88,-3.13,-4.25,-5.29,-6.73])/1000
#y_error = np.zeros_like(rel_spacing)
y_error = rel_spacing*0.01

#Preliminary plot of points
plt.scatter(m,rel_spacing,marker='x')
plt.xlabel('Minima Order (dimensionless)')
plt.ylabel('Minima Spacing (m)')


#Linear Least Squares Regression
fit_space,cov_space = np.polyfit(m,rel_spacing,1,w=1/y_error,cov=True)
sig_0 = np.sqrt(cov_space[0,0]) #The uncertainty in the slope
sig_1 = np.sqrt(cov_space[1,1]) #The uncertainty in the intercept
pSpace=np.poly1d(fit_space)

#Output relevant parameters
print('Slope = %.3e +/- %.3e' %(fit_space[0],sig_0))
print('Intercept = %.3e +/- %.3e' %(fit_space[1],sig_1))


#Plot fitted data

plt.plot(m,pSpace(m),color='#ED9121')
plt.grid()
plt.errorbar(m,rel_spacing,yerr=y_error,linestyle='none', mew=2, ms=3, capsize=0,color='black')
plt.show()

#Error propogation
sigma_lamda = 1e-09
sigma_d = 20e-06
sigma_f = 1e-03
lamda = 670e-09
f = 500e-03
d = 230e-06
sigma_s = np.sqrt((sigma_lamda*f/d)**2+(sigma_f*lamda/d)**2+(sigma_d*lamda*f/(d**2))**2)
print(sigma_s)

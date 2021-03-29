#Written by Jacob J. Edginton
import numpy as np
import matplotlib.image
from skimage import data
import matplotlib.pyplot as plt
from skimage.io import imread, imshow


#load image, different images can be loaded as long as they are in the correct folder, or the path is specified
image=imread('beam.jpg', as_gray=True) #RBG code generated by letting as_gray=False

#imshow(image) #you can use this to view the image you have loaded 


#Define xy plane of object bassed on image array 
y = image[500]
x=np.array(range(0,len(y)))


#Slice data to the linear region
w = y[(x<1600) & (x>300)]
z = x[(x<1600) & (x>300)]


#Fit data to a straight line
fit_im,cov_im = np.polyfit(z,w,1,cov=True)
sig_0 = np.sqrt(cov_im[0,0]) #The uncertainty in the slope
sig_1 = np.sqrt(cov_im[1,1]) #The uncertainty in the intercept
pSpace=np.poly1d(fit_im)


#Output relevant parameters
print('Slope = %.3e +/- %.3e' %(fit_im[0],sig_0))
print('Intercept = %.3e +/- %.3e' %(fit_im[1],sig_1))

#Plot fitted data
plt.xlabel('X - Position (cm)')
plt.plot(z,w,color='#ff781f')
plt.plot(z,pSpace(z))
plt.show()

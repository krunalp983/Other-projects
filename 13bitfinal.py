# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 02:31:16 2020

@author: krunal patel
"""


from scipy import signal

import matplotlib.pyplot as plt

import numpy as np

npts=1664

t = np.linspace(0, 10, npts, endpoint=True) #timevector



print(len(t))

y = np.repeat([1., 1., 1.,1., 1., -1.,-1.,1.,1.,-1.,1.,-1.,1.,],128)
plt.figure()
plt.plot(t,y,color='red')
plt.title('Data Sequence 1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
plt.grid('on')



#autocorrelation
syy = signal.correlate(y,y,mode='same')/128
#t2=-1*t[::-1]
#t2=np.append(t2)
plt.figure()
plt.plot(t,syy,color='blue')
plt.title('Barker Code- 13 bit length')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
plt.grid('on')
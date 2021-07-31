# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:51:17 2020

@author: krunal patel
"""


from scipy import signal

import matplotlib.pyplot as plt

import numpy as np

npts=512

t = np.linspace(0, 10, npts, endpoint=True) #timevector



print(len(t))

y = np.repeat([1., 1., 1., -1.,],128)
plt.figure()
plt.plot(t,y)
plt.title('Data Sequence 1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
plt.grid('on')

y1 = np.repeat([1., 1., -1., 1.,],128)
plt.figure()
plt.plot(t,y1)
plt.title('Signal 2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
plt.grid('on')

#autocorrelation
syy = signal.correlate(y,y,mode='same')/128
#t2=-1*t[::-1]
#t2=np.append(t2)
plt.figure()
plt.plot(t,syy)
plt.title('Barker Code- 4 bit complementry length')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
plt.grid('on')


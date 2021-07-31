# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:05:57 2020

@author: krunal
"""

# %reset -f

import scipy.io as spio
from scipy import signal
import os
import matplotlib.pyplot as plt
import numpy as np
# import datetime



DataPath='C:\\Users\\krunal patel\\Desktop\\summer semeser\\telecommunication\\python script\\project1\\'
# DataPath='C:/Toralf/Python/Uni_HWI/scripts20/RAW/'

Files=os.listdir(DataPath)

currentfile=str(DataPath)+str(Files[0])

# importing MATLAB mat file   (containing radar raw data)
mat = spio.loadmat(currentfile, squeeze_me=True)

datenums=mat['datenums']
ranges=mat['ranges']
data=mat['data']


# datenums ~ days since year 0
# here only the time is important for us -> hours, minutes, seconds
# => fraction / remainder of the integer

t=(datenums-np.floor(np.min(datenums)))*24



# number of range gates , data points, receivers
noRG=np.size(data,0)
noDP=np.size(data,1)
noRx=np.size(data,2)

RXsel=1


# averaged power per receiver for all datapoints

PWR=np.zeros([noRG,noRx])

for rx in range(noRx):
    PWR[:,rx]=20*np.log10(np.abs(np.mean(data[:,:,rx],1)))


plt.figure()
plt.plot(PWR,ranges)
plt.legend(['RX0','RX1','RX2','RX3','RX4','RX5','RX6','RX7','RX8'])
plt.xlabel('power /dB')
plt.ylabel('ranges /km')


# cross-correlation for one range and two receivers -> testing reasons


RG=9
x1=data[RG,:,2]
x2=data[RG,:,6]



# t=np.linspace(0,15,16)

# x1=[1,0,0,1,1,1,0,0,0,1,1,0,0,1,0,0,]
# x2=np.round(np.random.rand(16))

plt.figure()
plt.plot(t,np.abs(x1),t,np.abs(x2))

# xcor=signal.correlate(x1s,x2s,method='direct',mode='full')
xcor=signal.correlate(x1,x2,method='direct',mode='full')
# xcor2=np.correlate(a=x1,v=x2)
# xcor3=np.cov(x1,x2)
# xcor4=signal.correlate(np.real(x1),np.real(x2))+1j*signal.correlate(np.imag(x1),np.imag(x2))





# xcor=xcor/max(xcor)

tc=(t-min(t))*60*60
t2=-1*tc[::-1]
t2=np.append(t2, tc[1:])



plt.figure()
plt.subplot(2,1,1)
plt.plot(t2,abs(xcor))
plt.xlabel('tau')
plt.ylabel('abs(xcor)')
plt.subplot(2,1,2)
plt.plot(t2,np.angle(xcor)/np.pi*180)
plt.xlabel('tau')
plt.ylabel('phase /°')




# cross-correlation for all ranges and all receivers


Xcor=np.zeros([noRG,noDP*2-1,noRx])+1j*np.zeros([noRG,noDP*2-1,noRx])

for rg in range(noRG):
    Xcor[rg,:,0]=signal.correlate(data[rg,:,0],data[rg,:,1])
    Xcor[rg,:,1]=signal.correlate(data[rg,:,0],data[rg,:,2])
    Xcor[rg,:,2]=signal.correlate(data[rg,:,1],data[rg,:,2])



plt.figure()
for rx in range(noRx):
    plt.subplot(2,3,rx+1)
    plt.pcolor(t2,ranges,10*np.log10(np.abs(Xcor[:,:,rx])),cmap='jet')
    plt.xlabel('abs(xcor) /dB')
    plt.ylabel('range /km')
    plt.clim([20,90])
    plt.colorbar()
for rx in range(noRx):
    plt.subplot(2,3,rx+4)
    plt.pcolor(t2,ranges,np.angle(Xcor[:,:,rx])/np.pi*180,cmap='jet')
    plt.clim([-180, 180])
    plt.colorbar()
    plt.ylabel('range /km')
    plt.xlabel('phase /°')



# line plots for the mean of xcor (along the data points - time )

plt.figure()
plt.subplot(1,3,1)
plt.plot(PWR,ranges)
plt.legend(['Rx1','Rx2','Rx3','Rx4','Rx5','Rx6','Rx7','Rx8','Rx9'])
plt.xlabel('power /dB')
plt.grid('on')
plt.subplot(1,3,2)
plt.plot(10*np.log10(Xcor[:,noDP,:]),ranges)
plt.legend(['1-2','1-3','2-3','3-4','4-5','5-6','7-8','8-9','1-9','2-7','3-6'])
plt.xlabel('xcor ampl. /dB')
plt.grid('on')
plt.subplot(1,3,3)
plt.plot(np.angle(Xcor[:,noDP,:])/np.pi*180,ranges)
plt.legend(['1-2','1-3','2-3','3-4','4-5','5-6','7-8','8-9','1-9','2-7','3-6'])
plt.xlabel('phase / °')
plt.grid('on')





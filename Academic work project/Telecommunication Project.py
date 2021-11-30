# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:05:57 2020

@author: krunal
"""

# %reset -f

import scipy.io as spio
import os
import matplotlib.pyplot as plt
import numpy as np
# import datetime



DataPath='C:\\Users\\krunal patel\\Desktop\\summer semeser\\telecommunication\\python script\\project1\\'

Files=os.listdir(DataPath)

currentfile=str(DataPath)+str(Files[0])

# importing MATLAB mat file   (containing radar raw data
mat = spio.loadmat(currentfile, squeeze_me=True)

datenums=mat['datenums']
ranges=mat['ranges']
data=mat['data']
antpos=mat['antpos']
plt.figure()
plt.axes(projection='polar')
for k in range(len(antpos)):
    plt.plot(np.real(antpos[k]),np.imag(antpos[k]),'.')
    plt.xlabel('real')
    plt.ylabel('imag')

# datenums ~ days since year 0
# here only the time is important for us -> hours, minutes, seconds
# => fraction / remainder of the integer

t=(datenums-np.floor(np.min(datenums)))*24

# number of range gates , data points, receivers
noRG=np.size(data,0)
noDP=np.size(data,1)
noRx=np.size(data,2)


RXsel=1;


 #time series
plt.figure() #IQ diagram
y=data[0,:,RXsel]
plt.plot(np.real(y),np.imag(y),'.')
y=data[38,:,RXsel]
plt.plot(np.real(y),np.imag(y),'.')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('IQ Diagram')

for i in range(0):
    y=data[i,:,RXsel]
    
    # I/Q diagram
    plt.figure()
    plt.plot(np.real(y),np.imag(y),'.')
    plt.xlabel('real')
    plt.ylabel('imag')
    
    
# power of the complex valued voltages (measured by the receiver)
PWR=10*np.log10(np.abs(data[:,:,6]*data[:,:,6]))

PWR.shape
type(PWR)


plt.figure()
plt.pcolor(t,ranges,PWR)
plt.xlabel('time / HH')
plt.ylabel('range /km')
plt.title('power /dB')
plt.clim(20,70)
plt.colorbar()




# combine the data of all the receivers - complex sum !
pwr=np.zeros([noRG,noRx])
for rx in range(noRx):
    
    
    pwr[:,rx]=20*np.log10(np.abs(np.mean(data[:,:,rx],1)))
    pwr1=20*np.log10(np.abs(np.mean(data[:,:,1],1)))
    pwr2=20*np.log10(np.abs(np.mean(data[:,:,2],1)))
    pwr0=20*np.log10(np.abs(np.mean(data[:,:,0],1)))
    pwrsum=(pwr1+pwr2+pwr0)/3
#for rx in range(noRx):
#pwrcomb=20*np.log10(np.abs(np.mean((datacomb,1))))
#pwrcomb2=np.sum(pwr[:,1],pwr[:,2])
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(pwr,ranges)
    plt.xlabel('Power(dB)')
    plt.ylabel('Range(km)')
    plt.legend(['Rxr-1', 'Rxr-2','Rxr-3','Rxr-4','Rxr-5','Rxr-6','Rxr-7', 'Rxr-8','Rxr-9'])
    plt.title('Power profile for individual receivers')
    plt.subplot(1,2,2)
    plt.plot(pwrsum,ranges)
    plt.xlabel('Power(dB)')
    plt.ylabel('Range(km)')
    #plt.title('Combined Power profile of all receivers')
#power profile - lineplot
#plt.figure()
#plt.subplot(1,4,1)
#plt.plot(pwr2[:,0],ranges)
#plt.xlabel('Power(dB)')
#plt.ylabel('Range(km)')
#plt.title('Power profile - Rxr 1')
#plt.subplot(1,4,2)
#plt.plot(pwr1[:,0],ranges)
#plt.xlabel('Power(dB)')
#plt.ylabel('Range(km)')
#plt.title('Power profile - Rxr 2')
#plt.subplot(1,4,3)
#plt.plot(pwr0[:,0],ranges)
#plt.xlabel('Power(dB)')
#plt.ylabel('Range(km)')
#plt.title('Power profile - Rxr 2')
#plt.subplot(1,4,4)
#plt.plot(pwrcomb[:,0],ranges)
#plt.xlabel('Power(dB)')
#plt.ylabel('Range(km)')
#plt.title('Combined ower profile of all Rxrs')
#combined power
    datacomb=np.sum(data,2)
    pwrcomb=20*np.log10(np.abs(datacomb))
    plt.figure()
    plt.pcolor(t,ranges,pwrcomb)
    plt.xlabel('Time(hrs)')
    plt.ylabel('Range(km)')
    plt.title('Combined Power(dB)')
    plt.clim(20,90)
    plt.colorbar()

y=data[0,:,RXsel]

t=(datenums-np.floor(np.min(datenums)))*24

def make_fft(t,y):
    dt = t[1]-t[0] # dt -> temporal resolution ~ sample rate
    f = np.fft.fftfreq(t.size, dt) # frequency axis
    Y = np.fft.fft(y)   # FFT
    f=np.fft.fftshift(f)
    Y= np.fft.fftshift(Y)/(len(y))
    return f,Y



tsec=t*60*60
#No. Receiver 6
f,spec=make_fft(tsec,data[-25,:,6])


plt.figure()
plt.plot(f,10*np.log10(abs(spec)))
plt.grid ('on')
plt.xlabel('f / Hz')
plt.ylabel('amplitude /dB')




# Spectra for all ranges and all receivers


Spectr=np.zeros([noRG,noDP,noRx])+1j*np.zeros([noRG,noDP,noRx])

for rx in range(noRx):
    for rg in range(noRG):
        f,Spectr[rg,:,rx]=make_fft(tsec,data[rg,:,rx])

for rx in range(noRx):
    plt.figure()
    plt.pcolor(f,ranges,10*np.log10(abs(Spectr[:,:,rx]*abs(Spectr[:,:,rx]))))
    plt.clim([-15, 15])
    plt.xlabel('f /Hz')
    plt.ylabel('range /km')
    plt.colorbar()
#cross spectra range for all range and receivers






# Cross-Spectra for all ranges and all receivers

XSpectr=np.zeros([noRG,noDP,noRx])+1j*np.zeros([noRG,noDP,noRx])

XSpectr[:,:,0]=Spectr[:,:,0]*np.conj(Spectr[:,:,1])
XSpectr[:,:,1]=Spectr[:,:,0]*np.conj(Spectr[:,:,2])
XSpectr[:,:,2]=Spectr[:,:,1]*np.conj(Spectr[:,:,2])

# plt.figure()
# plt.pcolor(f,ranges,10*np.log10(abs(XSpectr[:,:,1]))/2)
# # plt.pcolor(f,ranges,np.angle(XSpectr[:,:,1]))
# plt.colorbar()
# plt.clim([-15, 15])


plt.figure()
for rx in range(noRx):
    plt.subplot(1,3,rx+1)
    plt.pcolor(f,ranges,10*np.log10(abs(XSpectr[:,:,rx]))/2,cmap='jet')
    plt.clim([-15, 15])
    plt.title('XSp ampl')

plt.figure()    
for rx in range(noRx):
    plt.subplot(1,3,rx+1)
    plt.pcolor(f,ranges,np.angle(XSpectr[:,:,rx])/np.pi*180,cmap='jet')
    plt.clim([-180, 180])
    # plt.colorbar()
    plt.title('XSp phase')
    




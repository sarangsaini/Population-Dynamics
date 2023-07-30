import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

tstop = 100000

dt = 0.1
t = np.arange(0,tstop,dt)
t = t/1000

x1 = np.zeros((len(t),1))
x2 = np.zeros((len(t),1))

rng = np.random.default_rng(2)	# to generate a vector of random numbers

I1 = 0.18+0.5*rng.normal(0,1,size=len(t))
I2 = -0.1+0.5*rng.normal(0,1,size=len(t))

W11 = -0.004*0
W22 = 0.1
W12 = 0.16
W21 = -0.5

D1 = 4
D2 = 7

for ii in range(10,len(t)):

  	x1[ii] = (-x1[ii-1]*W11*x1[ii-1]+I1[ii]+W21*x2[ii-D1])*dt/3+x1[ii-1]#+rng.normal(0,0.01,size=1)
  	x2[ii] = (-x2[ii-1]*W22*x2[ii-1]+I2[ii]+W12*x1[ii-D2])*dt/7+x2[ii-1]#+rng.normal(0,0.01,size=1)


plt.subplot(3,1,1)
plt.plot(t,x1)
plt.plot(t,x2)
plt.xlim([5,15])

fs = 10000
f2, t2, Zxx = signal.stft(x1.T-np.mean(x1), fs, window='hann', nperseg=4096*4, noverlap=1024)
b = np.squeeze(Zxx)
plt.subplot(3,1,2)
plt.pcolormesh(t2, f2, np.abs(b))
plt.title('Population 1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0,20])


f2, t2, Zxx = signal.stft(x2.T-np.mean(x2), fs, window='hann', nperseg=4096*4, noverlap=1024)
b = np.squeeze(Zxx)
plt.subplot(3,1,3)
plt.pcolormesh(t2, f2, np.abs(b))
plt.title('Population 2')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0,20])
plt.show()

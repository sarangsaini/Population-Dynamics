import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

tstop = 20000

dt = 0.1
t = np.arange(0,tstop,dt)
t = t/1000

x1 = np.zeros((len(t),1))
x2 = np.zeros((len(t),1))

rng = np.random.default_rng(2)	# to generate a vector of random numbers

I1 = 0.18+0.05*rng.normal(0,1,size=len(t))
I2 = 0.05+0.05*rng.normal(0,1,size=len(t))

W11 = 0.7
W22 = -0.25
W12 = 1.2
W21 = -1.3

D1 = int(1/dt)
D2 = int(3/dt)

for ii in range(int(5/dt),len(t)):
	
  x1[ii] = (-x1[ii-1]+W11*x1[ii-1]+I1[ii]+W21*x2[ii-D1])*dt/3+x1[ii-1]
  x2[ii] = (-x2[ii-1]+W22*x2[ii-1]+I2[ii]+W12*x1[ii-D2])*dt/6+x2[ii-1]


plt.subplot(3,1,1)
plt.plot(t,x1)
plt.plot(t,x2)
plt.xlim([0,tstop/1000])

fs = 10000
f2, t2, Zxx = signal.stft(x1.T-np.mean(x1), fs, window='hann', nperseg=1024*2, noverlap=1024/2)
b = np.squeeze(Zxx)
plt.subplot(3,1,2)
plt.pcolormesh(t2, f2, np.abs(b),shading='gouraud')
plt.title('Population 1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0,100])


f2, t2, Zxx = signal.stft(x2.T-np.mean(x2), fs, window='hann', nperseg=1024*2, noverlap=1024/2)
b = np.squeeze(Zxx)
plt.subplot(3,1,3)
plt.pcolormesh(t2, f2, np.abs(b),shading='gouraud')
plt.title('Population 2')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0,100])
plt.show()


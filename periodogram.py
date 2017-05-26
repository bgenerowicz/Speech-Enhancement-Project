"# coding = utf-8"
####
##    SETUP
###


import scipy.io
import numpy as np
import pylab

fs = 16e3
tsegment = 20e-3
sseg = tsegment * fs


#import matplotlib

#matplotlib.use("Agg")

import matplotlib.pyplot as plt

import wave
import soundfile as sf


###
## Read Data & Pad Data & Apply Window & FFT
###

#Read
newdata, fs = sf.read('Audio/clean.wav')


# Generate White Gaussian noise with variance 1
mean = 0
std = 0.1
num_samples = len(newdata)
wgn = np.random.normal(mean, std, num_samples)
data = newdata + wgn


# plt.plot(wgn)
# plt.show()
# plt.plot(newdata)
# plt.show()



rms = [block for block in sf.blocks('Audio/clean.wav', blocksize=320, overlap=160)]





#Pad
lastarr = len(rms) - 1  # take last array of the list
length = len(rms[lastarr])  # calculate length of last array
remainder = sseg - (length % sseg)  # calculate length of padding
rms[lastarr] = np.pad(rms[lastarr], (0, int(remainder)), 'constant')  # pad
rmsarray = np.vstack(rms)


#Filter with Hanning window

#Create & Apply hanning window
hanning_segment = np.hanning(rmsarray.shape[1])
#rmsarray_han = np.multiply(hanning_segment,rmsarray)
rmsarray_han=rmsarray
#FFT

F_data = np.fft.fft(rmsarray_han)


#Periodogram

from scipy import signal


f, Pxx_den = signal.periodogram(newdata, fs)
numframe = rmsarray.shape[0]  # number of rows in cut matrix

plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

bins=rmsarray.shape[1]
Per = np.empty([0,bins])
# alpha=np.empty(bins)
Q=np.empty(bins)

for j in range(0, numframe):
    f, Psd = signal.periodogram(rmsarray[j,:], fs, 'hanning', 639)
    Per = np.vstack((Per, Psd))
    alpha =1/(1+(np.divide(Q, Psd)-1)**2)
    Q = alpha*Q + (np.ones(bins)-alpha)*Psd





## Ex
from scipy.io.wavfile import write
write('dirty.wav',16000,data)



end=1
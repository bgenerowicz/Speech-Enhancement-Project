####
##    SETUP
###


import scipy.io
from scipy import signal
import numpy as np
import pylab

import sounddevice as sd

fs = 16e3
tsegment = 20e-3
sseg = tsegment * fs

import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt

import wave
import soundfile as sf




###
## Read Data & Pad Data & Apply Window & FFT
###

#Read
newdata, samplerate = sf.read('Audio/clean.wav')
rms = [block for block in sf.blocks('Audio/clean.wav', blocksize=320, overlap=160)]

#Pad
lastarr = len(rms) - 1  # take last array of the list
length = len(rms[lastarr])  # calculate length of last array
remainder = sseg - (length % sseg)  # calculate length of padding
rms[lastarr] = np.pad(rms[lastarr], (0, int(remainder)), 'constant',constant_values=0)  # pad
rmsarray = np.vstack(rms)

#Filter with Hanning window

#Create & Apply hanning window
hanning_segment = np.hanning(rmsarray.shape[1])
#rmsarray_han = np.multiply(hanning_segment,rmsarray)
rmsarray_han=rmsarray
#FFT

F_data = np.fft.fft(rmsarray_han)



###
## MS Method
###

L=1500/tsegment #segment
k=rmsarray_han.shape[1] #number of freq bins
Qprev=np.array(np.zeros(k))
R=rmsarray_han.shape[0]
result=np.empty(k)

for j in range(0, R-1):
    fourier_row=np.fft.fft(rmsarray_han[j,:]) #take fourier of row
    psd_row=np.absolute(fourier_row)**2 #psd of row
    #psd_row[psd_row==0]=np.nan
    #alpha=1/(1+(Qprev/psd_row-1)**2)
    alpha = 0.5
    onevector=np.array(np.ones(k)) #make onevector
    Q = alpha*Qprev + (onevector-alpha) * psd_row
    result=np.vstack((result,Q))
    Qprev=Q

###
## Add Noise
###
mean = 0
std = 0.1
num_samples = len(newdata)
wgn = np.random.normal(mean, std, num_samples)
data = newdata + wgn


####
## Reconstruct Data
###

#FFT

#IFFT
IF_data = np.fft.ifft(F_data)


rmsshortend = IF_data[1:IF_data.shape[0], 160:320]  # take only the second half of all columns, rows: second to last
numframe = rmsshortend.shape[0]  # number of rows in cut matrix
reconstruction = IF_data[0, :]  # take the whole first row, all columns

for j in range(0, numframe):
    reconstruction = np.hstack((reconstruction, rmsshortend[j, :]))  # add the halved rows


newdata = np.pad(newdata, (0, int(remainder)), 'constant',constant_values=0)  # pad to subtract
residual = newdata-reconstruction

#sf.write('new_file.ogg', reconstruction.imag, samplerate)

end=1
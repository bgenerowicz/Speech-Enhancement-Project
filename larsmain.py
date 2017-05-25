####
##    SETUP
###


import scipy.io
import numpy as np
import pylab

fs = 16e3
tsegment = 20e-3
sseg = tsegment * fs

<<<<<<< HEAD
#import matplotlib
=======
import matplotlib
>>>>>>> master
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
rms[lastarr] = np.pad(rms[lastarr], (0, int(remainder)), 'constant')  # pad
rmsarray = np.vstack(rms)

#Filter with Hanning window

#Create & Apply hanning window
hanning_segment = np.hanning(rmsarray.shape[1])
#rmsarray_han = np.multiply(hanning_segment,rmsarray)
rmsarray_han=rmsarray
#FFT

F_data = np.fft.fft(rmsarray_han)




###
## Reconstruct Data
###

#FFT

#IFFT
IF_data = np.fft.ifft(F_data)


#rmsshortend = rmsarray[1:rmsarray.shape[0], 160:320]  # take only the second half of all columns, rows: second to last
#numframe = rmsshortend.shape[0]  # number of rows in cut matrix
#reconstruction = rmsarray[0, :]  # take the whole first row, all columns

#for j in range(0, numframe):
#    reconstruction = np.hstack((reconstruction, rmsshortend[j, :]))  # add the halved rows



rmsshortend = IF_data[1:IF_data.shape[0], 160:320]  # take only the second half of all columns, rows: second to last
numframe = rmsshortend.shape[0]  # number of rows in cut matrix
reconstruction = IF_data[0, :]  # take the whole first row, all columns

for j in range(0, numframe):
    reconstruction = np.hstack((reconstruction, rmsshortend[j, :]))  # add the halved rows



# plt.plot(newdata,reconstruction)
# plt.plot(newdata)

newdata = np.pad(newdata, (0, int(remainder)), 'constant')  # pad to subtract
<<<<<<< HEAD
test = np.subtract(newdata, reconstruction)
plt.plot(reconstruction)
plt.show()
=======
residual = np.subtract(newdata, reconstruction)


#Plots
f, axarr = plt.subplots(3, sharex=True)
axarr[2].plot(residual)
axarr[2].set_title('Residual')
pylab.ylim([-0.5, 0.5])
axarr[1].plot(reconstruction)
axarr[1].set_title('Reconstructed')
axarr[0].plot(newdata)
axarr[0].set_title('Original')

plt.show()




>>>>>>> master

end=1

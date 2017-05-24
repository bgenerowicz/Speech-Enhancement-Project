import scipy.io
import numpy as np

fs = 16e3
tsegment = 20e-3
sseg = tsegment * fs

import matplotlib.pyplot as plt
# plt.plot(cleandatarow)
# plt.ylabel('some numbers')
# plt.show()

import wave
import soundfile as sf

newdata, samplerate = sf.read('Audio/clean.wav')
rms = [block for block in sf.blocks('Audio/clean.wav', blocksize=320, overlap=160)]

lastarr = len(rms) - 1  # take last array of the list
length = len(rms[lastarr])  # calculate length of last array
remainder = sseg - (length % sseg)  # calculate lenght of padding
rms[lastarr] = np.pad(rms[lastarr], (0, int(remainder)), 'constant')  # pad
rmsarray = np.vstack(rms)

rmsshortend = rmsarray[1:rmsarray.shape[0], 160:320]  # take only the second half of all columns, rows: second to last
numframe = rmsshortend.shape[0]  # number of rows in cut matrix
reconstruction = rmsarray[0, :]  # take the whole first row, all columns

for j in range(0, numframe):
    reconstruction = np.hstack((reconstruction, rmsshortend[j, :]))  # add the halved rows

# plt.plot(newdata,reconstruction)
# plt.plot(newdata)

newdata = np.pad(newdata, (0, int(remainder)), 'constant')  # pad to subtract
test = np.subtract(newdata, reconstruction)
plt.plot(test)


end=1

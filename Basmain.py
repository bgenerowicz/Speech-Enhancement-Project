import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from SingleMic import segment_overlap as s_o
from SingleMic import inverse_segment_overlap as i_s_o
import time
import pylab

start_time = time.time()

#Variables
tsegment = 20e-3 #20ms segment
overlap = 0.5

#Import data & fs
data, fs = sf.read('Audio/clean.wav')

# Calc
s_segment = int(tsegment * fs)
s_overlap = int(overlap * s_segment)
# pad data with zeros
remainder = s_segment - (len(data) % s_segment)
data_extended = np.ravel(np.asmatrix(np.pad(data, (0, int(remainder)), 'constant')))


#Segment & overlap the data
data_seg_over = s_o.segment_overlap(data_extended,s_segment,s_overlap)


#Create & Apply hanning window
hanning_segment = np.hanning(data_seg_over.shape[1])
data_han_seg_over = np.multiply(hanning_segment,data_seg_over)

#FFT
data_han_seg_over = data_seg_over
F_data = np.fft.fft(data_han_seg_over)

#do stuff


#IFFT
IF_data = np.fft.ifft(F_data)
IF_data_array = np.ravel(IF_data)

#Invert segmentation / overlap
reconstructed_data = i_s_o.inverse_segment_overlap(IF_data_array,len(data_extended),s_segment,s_overlap)

#Calculate Residual
residual = data_extended - reconstructed_data

#
# transform back into array
# x_array = np.ravel(x)
# x_truncarray = i_s_o.inverse_segment_overlap(x_array,len(data_extended),s_segment,s_overlap)
#
#
#
# #calculate difference between initial and reconstructed signals



#Plots
f, axarr = plt.subplots(3, sharex=True)
axarr[2].plot(residual)
axarr[2].set_title('Residual')
pylab.ylim([-0.5, 0.5])
axarr[1].plot(reconstructed_data)
axarr[1].set_title('Reconstructed')
axarr[0].plot(data_extended)
axarr[0].set_title('Original')

plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
end = 1





import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import segment_overlap as s_o
import inverse_segment_overlap as i_s_o
import time
import pylab
#test

start_time = time.time()

#Variables
tsegment = 20e-3 #20ms segment
overlap = 0.5

#Import data & fs
data, fs = sf.read('Audio/clean.wav')

# Add Noise
# mean = 0
# std = 0.05
# num_samples = len(data)
# wgn = np.random.normal(mean, std, num_samples)
# data = data + wgn


# Calculate segment and overlap
s_segment = int(tsegment * fs)
s_overlap = int(overlap * s_segment)

# pad data with zeros
remainder = s_segment - (len(data) % s_segment)
data_extended = np.ravel(np.asmatrix(np.pad(data, (0, int(remainder)), 'constant')))

#Create & Apply Hanning
num_frames = int((len(data_extended) - s_overlap) / s_overlap)
hanning_segment = np.hanning(s_segment)

data_seg_over = np.zeros([num_frames,s_segment]) #Init variables/matrix
s_start = 0
for i in range(0, num_frames):
    data_seg_over[i,:] = np.multiply(hanning_segment,data_extended[s_start:(s_start+s_segment)])
    s_start = s_start + s_overlap

#Do stuff


#Reconstruct data
reconstructed_data = np.zeros(len(data_extended)) #init matrix/ variables
s_start = 0
for i in range(0, num_frames):
    reconstructed_data[s_start:(s_start + s_segment)] = reconstructed_data[s_start:(s_start+s_segment)] + data_seg_over[i,:]
    s_start = s_start + s_overlap


#Calculate Residual
residual = data_extended - reconstructed_data

#Plots
f, axarr = plt.subplots(3, sharex=True)
axarr[2].plot(residual)
axarr[2].set_title('Residual')
pylab.ylim([-0.5, 0.5])
axarr[1].plot(reconstructed_data)
axarr[1].set_title('Reconstructed')
axarr[0].plot(data_extended)
axarr[0].set_title('Original')

end = 1
# #Segment and overlap data
#
#
# # #Create & Apply hanning window
# # calculate number of frames
# # l = int((len(data_extended) - s_overlap) / s_overlap)
# # k = data_extended.shape
#
# # hanning_segment = np.hanning(data_seg_over.shape[1])
# # data_seg_over = np.multiply(hanning_segment,data_seg_over)
#
# #FFT
# # data_han_seg_over = data_seg_over
# F_data = np.fft.fft(data_seg_over) #F_data is 3611 by 320 where each 320 = 1 frame (l) and each 3611 = 1 frequency bin (k)
#
#
# #do stuff
#
# #Init Q
# Qprev = np.zeros([F_data.shape[0]])
# Q = np.zeros([F_data.shape[0],F_data.shape[0]])
# alpha = 0.5
# # for k in range(0,s_segment): #For each freq bin
# #     print(k)
#
# for l in range(0,F_data.shape[1]): #For each time seg
#     # Q[0,l] = alpha*Q[0,l-1]
#     Qnew = alpha*Qprev + (1-alpha)*np.absolute(F_data[:,l])**2
#     Qprev = Qnew
# #For each frame l
#
#
# # fbin = F_data[:,1]
#
# #IFFT
# IF_data = np.fft.ifft(F_data)
# IF_data_array = np.ravel(IF_data)
#
# #Invert segmentation / overlap
# reconstructed_data = i_s_o.inverse_segment_overlap(IF_data_array,len(data_extended),s_segment,s_overlap)
#
# #Calculate Residual
# residual = data_extended - reconstructed_data
#
#
# #
# # transform back into array
# # x_array = np.ravel(x)
# # x_truncarray = i_s_o.inverse_segment_overlap(x_array,len(data_extended),s_segment,s_overlap)
# #
# #
# # #calculate difference between initial and reconstructed signals
#
#
#
# # #Plots
# # f, axarr = plt.subplots(3, sharex=True)
# # axarr[2].plot(residual)
# # axarr[2].set_title('Residual')
# # pylab.ylim([-0.5, 0.5])
# # axarr[1].plot(reconstructed_data)
# # axarr[1].set_title('Reconstructed')
# # axarr[0].plot(data_extended)
# # axarr[0].set_title('Original')
#
#
# # from scipy.io.wavfile import write
# # write('Audio/Saves/dirtyhan.wav',16000,reconstructed_data)
#
#
# print("--- %s seconds ---" % (time.time() - start_time))
# end = 1





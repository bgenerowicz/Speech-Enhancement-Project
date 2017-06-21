import bas_functions as b_f
import matplotlib.pyplot as plt
# Variables
tsegment = 20e-3  # 20ms segment
overlap = 0.5
filelocation = 'Audio/clean.wav'
noise = 0  #Add noise? 1 = noise


data, fs = b_f.import_data(filelocation,noise) #import data (possibly with noise)

data = data/max(data)

s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


framed_data = b_f.frame_data(data,fs,s_segment,s_overlap) #Frame data using hanning
fft_data = b_f.transform_data(framed_data) #FFT of every segment, # Step 1 of slide 76, lecture 1
##########
#Do stuff#
##########
y_k = b_f.bas_bartlett(framed_data) # Step 2 of slide 76, lecture 1

# f, axarr = plt.subplots(2, sharex=True)
# axarr[0].plot(fft_data[1000,:])
# axarr[1].plot(y_k[1000,:])
# plt.show()


s_PSD_est = b_f.signal_PSD_estimate(y_k) # Step 3 of slide 76, lecture 1


s_est = b_f.signal_estimate(s_PSD_est,fft_data) # Step 4 of slide 76, lecture 1


ifft_data = b_f.i_transform_data(s_est) #inverse transform data, # Step 5 of slide 76, lecture 1

reconstructed_data = b_f.overlap_add(ifft_data,len(data),s_segment, s_overlap) # Overlap & add
b_f.make_plot(data,reconstructed_data) # Make plots

# b_f.play_array(data,fs)
# b_f.play_array(reconstructed_data,fs)

end =1






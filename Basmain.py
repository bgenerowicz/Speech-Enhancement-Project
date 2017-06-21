import bas_functions as b_f

# Variables
tsegment = 20e-3  # 20ms segment
overlap = 0.5
filelocation = 'Audio/clean.wav'
noise = 1  #Add noise? 1 = noise


data, fs = b_f.import_data(filelocation,noise) #import data (possibly with noise)

data = data/max(data)

s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


framed_data = b_f.frame_data(data,fs,s_segment,s_overlap) #Frame data using hanning
fft_data = b_f.transform_data(framed_data) #FFT of every segment
##########
#Do stuff#
##########
y_k = b_f.bas_bartlett(framed_data)

s_PSD_est = b_f.signal_PSD_estimate(y_k)


s_est = b_f.signal_estimate(s_PSD_est,fft_data)


ifft_data = b_f.i_transform_data(s_est) #inverse transform data
reconstructed_data = b_f.overlap_add(ifft_data,len(data),s_segment, s_overlap) # Overlap & add
b_f.make_plot(data,reconstructed_data) # Make plots

# b_f.play_array(data,fs)
# b_f.play_array(reconstructed_data,fs)

end =1






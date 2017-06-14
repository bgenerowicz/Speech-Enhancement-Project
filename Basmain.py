import bas_functions as b_f

# Variables
tsegment = 20e-3  # 20ms segment
overlap = 0.5




data, fs = b_f.import_data() #import data (possibly with noise)


s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


framed_data = b_f.frame_data(data,fs,s_segment,s_overlap) #Frame data using hanning
fft_data = b_f.transform_data(framed_data) #FFT of every segment
##########
#Do stuff#
##########
ifft_data = b_f.i_transform_data(fft_data) #inverse transform data
reconstructed_data = b_f.overlap_add(ifft_data,len(data),s_segment, s_overlap) # Overlap & add
b_f.make_plot(data,reconstructed_data) # Make plots


end =1






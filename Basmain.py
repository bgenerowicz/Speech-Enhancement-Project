import bas_functions as b_f
import larsfunctions as l_f



# Variables
tsegment = 20e-3  # 20ms segment
overlap = 0.5
filelocation = 'Audio/clean.wav'
noise = 'Audio/n.wav'


data,noise_only, fs = b_f.import_data(filelocation,noise) #import data (possibly with noise)


s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


framed_data = b_f.frame_data(data,fs,s_segment,s_overlap) #Frame data using hanning
y_k = b_f.transform_data(framed_data) #FFT of every segment, # Step 1 of slide 76, lecture 1

# Appling slide 76, lecture 1
# Py = b_f.bas_bartlett(framed_data) # Step 2 of slide 76, lecture 1
# Ps_est = b_f.signal_PSD_estimate(Py) # Step 3 of slide 76, lecture 1
# Sk_est = b_f.signal_estimate(Ps_est,y_k) # Step 4 of slide 76, lecture 1
# ifft_data = b_f.i_transform_data(Sk_est) #inverse transform data, # Step 5 of slide 76, lecture 1

# Applying slide 17, lecture 4
Py = b_f.bas_bartlett(framed_data) #
# Py = b_f.temp_psd(framed_data)



windowlength = int(1.5 / tsegment)
Pn_est = l_f.calculate_noisepsd_min(Py,tsegment,windowlength)

# Sk_est = b_f.wiener(Py,Pn_est,y_k)
# ifft_data = b_f.i_transform_data(Sk_est) #inverse transform data
#
# reconstructed_data = b_f.overlap_add(ifft_data,len(data),s_segment, s_overlap) # Overlap & add
# b_f.make_plot(data,reconstructed_data) # Make plots

#Testing
framed_noise = b_f.frame_data(noise_only,fs,s_segment,s_overlap)
b_f.plot_minPtrack(framed_noise,Pn_est,fs)

# b_f.play_array(data,fs)
# b_f.play_array(reconstructed_data,fs)

end =1






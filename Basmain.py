import bas_functions as b_f
import larsfunctions as l_f
import Noise_MMSE as c_f




# Variables
tsegment = 20e-3  # 20ms segment
overlap = 0.5
filelocation_clean = 'Audio/clean.wav'
filelocation_noise = 'Audio/n.wav'


s_n, fs = b_f.import_data(filelocation_clean) #import signal
n_n, fs = b_f.import_data(filelocation_noise) #import noise

n_n = 30*n_n  #Scale the noise
y_n = s_n + n_n #Add scaled noise to signal
s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)
windowlength = int(1.5 / tsegment)




# b_f.make_plot(y_n,n_n) # Make plots
s_n_framed = b_f.frame_data(s_n,fs,s_segment,s_overlap)
n_n_framed = b_f.frame_data(n_n,fs,s_segment,s_overlap) #Frame data using hanning
y_n_framed = b_f.frame_data(y_n,fs,s_segment,s_overlap)




y_k = b_f.transform_data(y_n_framed) #FFT of every segment

Py_true = b_f.temp_psd(y_n_framed)
Pn_true = b_f.temp_psd(n_n_framed) #Pn using exact noise


Pn_est = c_f.Noise_MMSE(y_n_framed,s_segment,y_k)
# Pn_est = l_f.calculate_noisepsd_min(Py_true,tsegment,windowlength)


# Py_est = b_f.bas_bartlett(y_n_framed) # Use bartlett to estimate Py



# b_f.make_recon_plot(Py_true[:,100],Py_est[:,100]) # Make plots


Sk_est = b_f.wiener(Py_true,Pn_est,y_k)

ifft_data = b_f.i_transform_data(Sk_est) #inverse transform data
reconstructed_data = b_f.overlap_add(ifft_data,len(y_n),s_segment, s_overlap) # Overlap & add

# snr = b_f.bas_SNR(y_n_framed,ifft_data, s_n, reconstructed_data,y_n,fs) #wrong
#SNR?
dSNR1 = b_f.bas_SNR(s_n_framed,y_n_framed)   #SNR of noisy signal before recon
dSNR2 = b_f.bas_SNR(s_n_framed,ifft_data)
b_f.plot_SNR(dSNR1,dSNR2) # Make plots


b_f.make_recon_plot(y_n,reconstructed_data) # Make plots


end = 1
# s_segment = int(tsegment * fs) # Calculate segment and overlap
# s_overlap = int(overlap * s_segment)
#
# framed_noise = b_f.frame_data(noise_only,fs,s_segment,s_overlap)
# framed_data = b_f.frame_data(data,fs,s_segment,s_overlap) #Frame data using hanning
# y_k = b_f.transform_data(framed_data) #FFT of every segment, # Step 1 of slide 76, lecture 1

# Appling slide 76, lecture 1
# Py = b_f.bas_bartlett(framed_data) # Step 2 of slide 76, lecture 1
# Ps_est = b_f.signal_PSD_estimate(Py) # Step 3 of slide 76, lecture 1
# Sk_est = b_f.signal_estimate(Ps_est,y_k) # Step 4 of slide 76, lecture 1
# ifft_data = b_f.i_transform_data(Sk_est) #inverse transform data, # Step 5 of slide 76, lecture 1

# Applying slide 17, lecture 4
# Py = b_f.bas_bartlett(framed_data) #
# Py_not ,Pn = b_f.temp_psd(framed_data,framed_noise)



# windowlength = int(1.5 / tsegment)
# Pn_est = l_f.calculate_noisepsd_min(Py,tsegment,windowlength)
# Pnest = b_f.Bas_min_ptrack(Py,fs)

# Sk_est = b_f.wiener(Py,Pn,y_k)
# ifft_data = b_f.i_transform_data(Sk_est) #inverse transform data
#
# reconstructed_data = b_f.overlap_add(ifft_data,len(data),s_segment, s_overlap) # Overlap & add
# b_f.make_plot(data,reconstructed_data) # Make plots

#Testing

# b_f.plot_minPtrack(framed_noise,Pn_est,fs)

# b_f.play_array(data,fs)
# b_f.play_array(reconstructed_data,fs)

# end =1






import frame_data as f_d
import import_data as i_d
import transform_data as t_d
import i_transform_data as i_t_d
import overlap_add as o_a
import make_plot as m_p

# Variables
tsegment = 20e-3  # 20ms segment
overlap = 0.5




data, fs = i_d.import_data() #import data (possibly with noise)


s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


framed_data = f_d.frame_data(data,fs,s_segment,s_overlap) #Frame data using hanning
fft_data = t_d.transform_data(framed_data) #FFT of every segment
##########
#Do stuff#
##########
ifft_data = i_t_d.i_transform_data(fft_data) #inverse transform data
reconstructed_data = o_a.overlap_add(ifft_data,len(data),s_segment, s_overlap) # Overlap & add
m_p.make_plot(data,reconstructed_data) # Make plots


end =1






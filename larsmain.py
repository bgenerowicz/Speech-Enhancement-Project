import scipy.io
from scipy import signal
import soundfile as sf
import numpy as np
import pylab
import sounddevice as sd
import matplotlib
import matplotlib.pyplot as plt
import wave
import time
import copy


from bas_functions import import_data
from bas_functions import frame_data
from bas_functions import transform_data
from larsfunctions import bartlett
from larsfunctions import calculate_noisepsd_min
from larsfunctions import Noise_MMSE
from larsfunctions import wiener
from bas_functions import i_transform_data
from larsfunctions import ml_estimation
from larsfunctions import dd_approach
from bas_functions import overlap_add
from bas_functions import play_array
from bas_functions import bas_SNR



tsegment = 20e-3
filelocation = 'Audio/clean+n.wav'
windowlength = int(1.5 / tsegment)  # segment in seconds to find minimum psd, respectively psd of noise
overlap = 0.5
#newdata,fs= import_data(filelocation)
filelocation_clean = 'Audio/clean.wav'
newdata_clean,_= import_data(filelocation_clean)
noise_location = 'Audio/n.wav'
alpha_smooth_exponential=0.85
alpha_smooth_dd=0.96
eps=0.0008


## Import Data
signal, fs = import_data(filelocation) #import signal
signal_clean, _ = import_data(filelocation_clean) #import clean signal
signal_noise,_ = import_data(noise_location) #import noise


s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


## Applying Framing and Hanning
signal_han = frame_data(signal,fs,s_segment,s_overlap)
signal_clean_han = frame_data(signal_clean,fs,s_segment,s_overlap)
signal_noise_han = frame_data(signal_noise,fs,s_segment,s_overlap)


## FT and Signal Power
F_data=transform_data(signal_han)
psd_F_data=np.absolute(F_data)**2

F_data_noise = transform_data(signal_noise_han)
psd_F_data_noise = np.absolute(F_data_noise)**2


## Smoothing PSD
F_data_bartlett = bartlett(psd_F_data) #bartlett smoothing
#F_data_exponential = exponentialsmoother(psd_F_data,alpha_smooth_exponential) #exponential smoothing
F_data_smoothed = copy.copy(F_data_bartlett)


## Calculate Noise PSD
noisevariance_min,alphastack = calculate_noisepsd_min(F_data_smoothed,tsegment,windowlength) #calculate Pn with minimum tracking
noisevariance_mmse = Noise_MMSE(signal_han,F_data,s_segment)
noisevariance = noisevariance_min


## Apply Wiener Gain
s_est = wiener(F_data_bartlett,noisevariance,F_data)
ifft_data = i_transform_data(s_est)
reconstructed_data = overlap_add(ifft_data,len(signal),s_segment,s_overlap)


## ML and DD Approach
sigma_s_ml = ml_estimation(F_data_smoothed,noisevariance)
sigma_s_dd = dd_approach(sigma_s_ml,noisevariance,F_data_smoothed,alpha_smooth_dd,eps)


## Calculate Wiener Gain
noisevariance[noisevariance == 0] = np.nan
gainmatrix = sigma_s_dd / (sigma_s_dd + noisevariance)
gainmatrix[np.isnan(gainmatrix) ] = 1
gainmatrix = np.maximum(gainmatrix,0)

## Apply Gain
s_est_min = F_data_smoothed * gainmatrix


## IFFT & Reconstruct
ifft_data = i_transform_data(s_est_min)
reconstruction = overlap_add(ifft_data,len(signal),s_segment,s_overlap)

## Calculate SNR
snr_a=bas_SNR(signal_clean_han,signal_han)
snr_a[np.isnan(snr_a)]=0

snr_b=bas_SNR(signal_clean_han,ifft_data)
snr_b[np.isnan(snr_b)]=0



## PLOTS

y=10*np.log10(psd_F_data_noise[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
signalpowerplot=plt.plot(x_axis2,y,color = 'g',alpha=0.4, label="Noise Power")

y=10*np.log10(F_data_smoothed[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
exponential_plot=plt.plot(x_axis2,y,color = 'g',alpha=0.9,  label="PSD Exponential Smoothed")

y=10*np.log10(noisevariance_min[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
noisevarianceplot=plt.plot(x_axis2,y,color = 'r',alpha=0.8, label="Noise Variance (Minimum)")

y=10*np.log10(noisevariance_mmse[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
noisevariance_mmse=plt.plot(x_axis2,y,color = 'r',alpha=0.5,  label="Noise Variance (MMSE)")



plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)


#
# ## Plots for Exponential Smoothing
# y=10*np.log10(psd_F_data[:,120])
# y[y == 0] = np.nan
# x_axis2 = 320*np.array(range(0,y.size))/fs
# signalpowerplot=plt.plot(x_axis2,y,color = 'c',alpha=0.4, label="Noise Power")
#
# y=10*np.log10(F_data_smoothed[:,120])
# y[y == 0] = np.nan
# x_axis2 = 320*np.array(range(0,y.size))/fs
# signalpowerplot=plt.plot(x_axis2,y,color = 'r',alpha=0.5, label="Noise Power")

plt.show()

end=1
import scipy.io
from scipy import signal
import soundfile as sf
import numpy as np
import pylab
import sounddevice as sd
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wave
import time

from larsfunctions import import_data
from larsfunctions import import_frame_data
from larsfunctions import transform_data
#from larsfunctions import i_transform_data
#from larsfunctions import overlap_add
from larsfunctions import calculate_residual
from larsfunctions import calculate_noisepsd_min
from larsfunctions import calculate_speechpsd_heuristic
from larsfunctions import calculate_wiener_gain
from larsfunctions import calculate_noisepsd_min_costas
from larsfunctions import wiener
from larsfunctions import Noise_MMSE
from larsfunctions import lars_bartlett
from larsfunctions import lars_exponentialsmoother


from bas_functions import bas_bartlett
from bas_functions import i_transform_data
from bas_functions import overlap_add
#from bas_functions import import_data
from bas_functions import frame_data
from bas_functions import transform_data

tsegment = 20e-3
filelocation = 'Audio/clean+n.wav'
windowlength = int(1.5 / tsegment)  # segment in seconds to find minimum psd, respectively psd of noise
overlap = 0.5
newdata,fs= import_data(filelocation)
noise = 'Audio/n.wav'
alpha_smooth=0.85


rmsarray_han,fs,remainder = import_frame_data(filelocation,tsegment)

s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)

F_data=transform_data(rmsarray_han)
psd_F_data=np.absolute(F_data)**2

F_data_bartlett=lars_bartlett(psd_F_data) #bartlett smoothing
F_data_exponential=lars_exponentialsmoother(psd_F_data,alpha_smooth) #exponential smoothing

noisevariance_min=calculate_noisepsd_min(F_data_exponential,tsegment,windowlength) #calculate Pn with minimum tracking

noisevariance_mmse=Noise_MMSE(rmsarray_han,F_data,s_segment)

s_est,gainmatrix=wiener(F_data_bartlett,psd_F_data,F_data)

ifft_data = i_transform_data(s_est)

reconstructed_data = overlap_add(ifft_data,len(newdata),s_segment,s_overlap)

remainder=np.empty(10)
residual = calculate_residual(filelocation,reconstructed_data,remainder)

s_est_mmse,gainmatrix=wiener(F_data,psd_F_data,F_data)

ifft_data = i_transform_data(s_est_mmse)

reconstructed_data_mmse = overlap_add(ifft_data,len(newdata),s_segment,s_overlap)

#sf.write('new_file3.ogg', reconstructed_data, fs)

sf.write('new_file3_mmse.ogg', reconstructed_data, fs)

y=10*np.log10(psd_F_data[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
signalpowerplot=plt.plot(x_axis2,y,color = 'beige')


y=10*np.log10(noisevariance_min[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
noisevarianceplot=plt.plot(x_axis2,y,color = 'black')

y=10*np.log10(noisevariance_mmse[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
noisevariance_mmse.plot=plt.plot(x_axis2,y,color = 'blue')


y=10*np.log10(F_data_exponential[:,180])
y[y == 0] = np.nan
x_axis2 = 320*np.array(range(0,y.size))/fs
exponential_plot=plt.plot(x_axis2,y,color = 'crimson')

plt.show()

end=1
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

from bas_functions import bas_bartlett
from bas_functions import i_transform_data
from bas_functions import overlap_add

tsegment = 20e-3
filelocation = 'Audio/clean+intersect.wav'
windowlength = int(1.5 / tsegment)  # segment in seconds to find minimum psd, respectively psd of noise
overlap = 0.5
newdata,fs= import_data(filelocation)

s_segment = int(tsegment * fs) # Calculate segment and overlap
s_overlap = int(overlap * s_segment)


rmsarray_han,fs,remainder = import_frame_data(filelocation,tsegment)

F_data=transform_data(rmsarray_han)
psd_F_data=np.absolute(F_data)**2
F_data_bartlett=bas_bartlett(F_data) #Bartlett Estimate, is PSD now

noisevariance=calculate_noisepsd_min(F_data_bartlett,tsegment,windowlength)

##TODO there are negative gains, where are they coming from?

speechpsd=calculate_speechpsd_heuristic(F_data,noisevariance)

s_est,gainmatrix=wiener(F_data_bartlett,noisevariance,F_data)

ifft_data = i_transform_data(s_est)

##TODO change overlapp add (hanning)
reconstructed_data =overlap_add(ifft_data,len(newdata),s_segment,s_overlap)

residual = calculate_residual(filelocation,reconstructed_data,remainder)

#sf.write('new_file2.ogg', reconstructed_data, fs)

end=1
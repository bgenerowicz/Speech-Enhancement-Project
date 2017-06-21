import scipy.io
from scipy import signal
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
from larsfunctions import i_transform_data
from larsfunctions import overlap_add
from larsfunctions import calculate_residual
from larsfunctions import calculate_noisepsd_min
from larsfunctions import calculate_speechpsd_heuristic
from larsfunctions import calculate_wiener_gain
from larsfunctions import calculate_noisepsd_min_costas
from larsfunctions import wiener
from bas_functions import bas_bartlett

tsegment = 20e-3
filelocation = 'Audio/clean.wav'
windowlength = int(1.5 / tsegment)  # segment in seconds to find minimum psd, respectively psd of noise

newdata,_=import_data(filelocation)
rmsarray_han,fs,remainder=import_frame_data(filelocation,tsegment)

F_data=transform_data(rmsarray_han)
psd_F_data=np.absolute(F_data)**2
F_data_bartlett=bas_bartlett(F_data) #Bartlett Estimate, is PSD now

noisevariance=calculate_noisepsd_min(F_data_bartlett,tsegment,windowlength)

##TODO there are negative gains, where are they coming from?


start_time = time.time()
speechpsd=calculate_speechpsd_heuristic(F_data,noisevariance)
#wienergain=calculate_wiener_gain(speechpsd,noisevariance)
s_est,gainmatrix=wiener(F_data_bartlett,noisevariance,F_data)
print("--- %s seconds for main---" % (time.time() - start_time))

##TODO multiply F_data matrix with gainmatrix (elementwise)


IF_data = i_transform_data(F_data)

##TODO change overlapp add (hanning)
reconstruction = overlap_add(IF_data)
residual = calculate_residual(filelocation,reconstruction,remainder)

#sf.write('new_file.ogg', reconstruction.imag, samplerate)

end=1
import scipy.io
from scipy import signal
import numpy as np
import pylab
import sounddevice as sd
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wave

from larsfunctions import import_data
from larsfunctions import import_frame_data
from larsfunctions import transform_data
from larsfunctions import i_transform_data
from larsfunctions import overlap_add
from larsfunctions import calculate_residual
from larsfunctions import calculate_noisepsd_min
from larsfunctions import calculate_speechpsd_heuristic
from larsfunctions import calculate_wiener_gain


tsegment = 20e-3
filelocation = 'Audio/clean.wav'



###
## Read Data & Pad Data & Apply Window & FFT
###

newdata,_=import_data(filelocation)
rmsarray_han,fs,remainder=import_frame_data(filelocation,tsegment)

F_data=transform_data(rmsarray_han)

noisevariance=calculate_noisepsd_min(F_data,tsegment)


# ## Calculate Speech PSD with the Heuristic Approach of Lect.1+4
## Calculate the Gain with the Wiener Formula
## TODO there are negative gains, where are they coming from?
#

speechpsd=calculate_speechpsd_heuristic(F_data)
wienergain=calculate_wiener_gain(speechpsd,noisevariance)


## TODO multiply F_data matrix with gainmatrix (elementwise)




#IFFT
IF_data = i_transform_data(F_data)

####
## Reconstruct Data



reconstruction = overlap_add(IF_data)
residual = calculate_residual(filelocation,reconstruction,remainder)

#sf.write('new_file.ogg', reconstruction.imag, samplerate)

end=1
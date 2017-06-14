import numpy as np

def transform_data(framed_data):
    F_data = np.fft.fft(framed_data)
    return F_data

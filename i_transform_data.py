import numpy as np

def i_transform_data(F_data):
    # IFFT
    IF_data = np.fft.ifft(F_data)

    return IF_data
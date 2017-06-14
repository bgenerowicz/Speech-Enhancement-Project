import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pylab

def frame_data(data,fs,s_segment,s_overlap):

    # pad data with zeros
    remainder = s_segment - (len(data) % s_segment)
    data_extended = np.ravel(np.asmatrix(np.pad(data, (0, int(remainder)), 'constant')))

    # Create & Apply Hanning to segment & overlap data
    num_frames = int((len(data_extended) - s_overlap) / s_overlap)
    hanning_segment = np.hanning(s_segment)

    data_seg_over = np.zeros([num_frames, s_segment])  # Init variables/matrix
    s_start = 0
    for i in range(0, num_frames):
        data_seg_over[i, :] = np.multiply(hanning_segment, data_extended[s_start:(s_start + s_segment)])
        s_start = s_start + s_overlap

    return data_seg_over

def i_transform_data(F_data):
    # IFFT
    IF_data = np.fft.ifft(F_data)

    return IF_data



def import_data():
    # Import data & fs
    data, fs = sf.read('Audio/clean.wav')

    # Add Noise
    # mean = 0
    # std = 0.05
    # num_samples = len(data)
    # wgn = np.random.normal(mean, std, num_samples)
    # data = data + wgn

    return data, fs



def make_plot(data,reconstructed_data):
    # Calculate Residual
    residual = data - reconstructed_data

    # Plots
    f, axarr = plt.subplots(3, sharex=True)
    axarr[2].plot(residual)
    axarr[2].set_title('Residual')
    pylab.ylim([-0.5, 0.5])
    axarr[1].plot(reconstructed_data)
    axarr[1].set_title('Reconstructed')
    axarr[0].plot(data)
    axarr[0].set_title('Original')

    plt.show()

def overlap_add(IF_data,len_data,s_segment,s_overlap):


    num_frames = int((len_data - s_overlap) / s_overlap)



    reconstructed_data = np.zeros(len_data)  # init matrix/ variables
    s_start = 0
    for i in range(0, num_frames):
        reconstructed_data[s_start:(s_start + s_segment)] = reconstructed_data[s_start:(s_start + s_segment)] + IF_data[
                                                                                                                i, :]
        s_start = s_start + s_overlap

    return reconstructed_data

def transform_data(framed_data):
    F_data = np.fft.fft(framed_data)
    return F_data
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pylab
import cmath as cm
import sounddevice as sd

def frame_data(data,fs,s_segment,s_overlap):
    #frame_data takes data, segments it and applies a hanning window

    # pad data with zeros so that the segments fit
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
    # Step 5 of slide 76, lecture 1
    data = np.fft.ifft(F_data)
    IF_data = np.real(data) #imag parts are almost zero, as should be
    return IF_data



def import_data(filelocation,noise):
    # Import data & fs
    data, fs = sf.read(filelocation)   #Import clean signal
    n, fs = sf.read(noise)     #Import noise
    noisy_data = data+20*n      #Add noise

    noisy_data = noisy_data / max(noisy_data)  #normalize

    return noisy_data, fs



def make_plot(data,reconstructed_data):
    residual = data - reconstructed_data

    # Plots
    f, axarr = plt.subplots(3, sharex=True)
    axarr[2].plot(residual)
    axarr[2].set_title('Residual')
    pylab.ylim([-1, 1])
    axarr[1].plot(reconstructed_data)
    axarr[1].set_title('Reconstructed')
    axarr[0].plot(data)
    axarr[0].set_title('Original')

    plt.show()

def overlap_add(IF_data,len_data,s_segment,s_overlap):
    #The goal is to take the ifft'd data and overlap + add

    num_frames = int((len_data - s_overlap) / s_overlap)



    reconstructed_data = np.zeros(len_data)  # init matrix/ variables
    s_start = 0
    for i in range(0, num_frames):
        reconstructed_data[s_start:(s_start + s_segment)] = reconstructed_data[s_start:(s_start + s_segment)] + IF_data[
                                                                                                                i, :]
        s_start = s_start + s_overlap
    reconstructed_data = reconstructed_data / max(reconstructed_data) #normalize
    return reconstructed_data

def transform_data(framed_data):
    #Apply a fft to the data
    # Step 1 of slide 76, lecture 1
    F_data = np.fft.fft(framed_data)

    return F_data

def bas_bartlett(framed_data):
    # Bartlett: split a time segment of 320 into smaller segments, take psd of smaller segments (make length 320 again)
    # and average over each of the segments
    # Step 2 of slide 76, lecture 1

    bartlett_samples = 20
    bartlett_frames = int(framed_data.shape[1] / bartlett_samples)
    F_data = np.zeros(framed_data.shape)

    for j in range(0, framed_data.shape[0]):
        for i in range(0, bartlett_frames):
            F_data[j, :] = F_data[j, :] + np.absolute(np.fft.fft(framed_data[j, bartlett_samples * i:bartlett_samples * (i + 1)], framed_data.shape[1])) ** 2
        F_data[j, :] = F_data[j, :] / framed_data.shape[1]
    return F_data

def signal_PSD_estimate(y_k):
    # Calculate the estimated signal PSD
    # Step 3 of slide 76, lecture 1

    s_PSD_est = np.zeros(y_k.shape)
    for i in range(0,y_k.shape[0]):
        s_PSD_est[i,:] = np.maximum(y_k[i,:],0.2*max(y_k[i,:]))

    return s_PSD_est


def signal_estimate(s_PSD_est,fft_data):
    # Calculate the estimate of the signal in frequency domain
    # Step 4 of slide 76, lecture 1
    s_est = np.array(np.zeros(fft_data.shape),dtype='complex_')

    angles = np.angle(fft_data)

    for i in range(0,fft_data.shape[0]):
        phase = np.exp(np.multiply(1j,angles[i,:]))
        magnitude = np.sqrt(s_PSD_est[i,:])
        s_est[i,:] = np.multiply(magnitude,phase)

    return s_est

def play_array(data,fs):
    data = data / max(data)
    sd.play(data,fs)

def wiener(Py,Pn,y_k):
    Py[Py == 0] = np.nan  # set nan to avoid division by zero


    H = (1- Pn/Py) # Work out wiener gain

    H[np.isnan(H)] = 0  #convert nans back to zeros
    Sk_est = H*y_k

    return Sk_est
import soundfile as sf
import numpy as np


def import_data(filelocation):

    data, fs = sf.read(filelocation)

    return data, fs


def import_frame_data(filelocation,tsegment):

    _, samplerate = sf.read(filelocation)
    rms = [block for block in sf.blocks(filelocation, blocksize=320, overlap=160)]

    sseg = tsegment * samplerate

    # Pad
    lastarr = len(rms) - 1  # take last array of the list
    length = len(rms[lastarr])  # calculate length of last array
    remainder = sseg - (length % sseg)  # calculate length of padding
    rms[lastarr] = np.pad(rms[lastarr], (0, int(remainder)), 'constant', constant_values=0)  # pad
    rmsarray = np.vstack(rms)

    # # Add Noise
    # mean = 0
    # std = 0.2
    # num_samples = rms.shape[0]
    # wgn = np.random.normal(mean, std, num_samples)
    # newdata = newdata + wgn

    # Create & Apply hanning window
    hanning_segment = np.hanning(rmsarray.shape[1])
    # rmsarray_han = np.multiply(hanning_segment,rmsarray)
    rmsarray_han = rmsarray  # for testing

    return rmsarray_han,samplerate,remainder


def transform_data(framed_data):

    F_data = np.fft.fft(framed_data)

    return F_data


def i_transform_data(F_data):

    IF_data = np.fft.ifft(F_data)

    return IF_data


def overlap_add(IF_data):

    rmsshortend = IF_data[1:IF_data.shape[0], 160:320]  # take only the second half of all columns, rows: second to last
    numframe = rmsshortend.shape[0]  # number of rows in cut matrix
    reconstruction = IF_data[0, :]  # take the whole first row, all columns

    for j in range(0, numframe):
        reconstruction = np.hstack((reconstruction, rmsshortend[j, :]))  # add the halved rows

    return reconstruction


def calculate_residual(filelocation,reconstruction,remainder):

    data, fs = sf.read(filelocation)
    data = np.pad(data, (0, int(remainder)), 'constant', constant_values=0)  # pad to subtract

    residual = data - reconstruction

    return residual


def calculate_noisepsd_min(F_data,tsegment):


    ###
    ## MS Method (Hendricks Book eq.(6.2) + Martin2001 paper in Dropbox)
    ###
    ## TODO The Bias correction for alpha is missing!


    k = F_data.shape[1]  # number of freq bins
    Qprev = np.array(np.zeros(k))
    R = F_data.shape[0]
    noisevariance = np.empty(k)
    gainmatrix = np.empty(k)

    for j in range(0, R - 1):
        fourier_row = F_data[j, :]  # load fourier of row
        psd_row = np.absolute(fourier_row) ** 2  # psd of row
        psd_row[psd_row == 0] = np.nan  # set nan to avoid division by zero
        alpha = 1 / (1 + (Qprev / psd_row - 1) ** 2)  # calculate alpha, see paper in dropbox
        # alpha = 0.85 #for testing

        onevector = np.array(np.ones(k))  # make onevector
        Q = alpha * Qprev + (onevector - alpha) * psd_row  # hendricksbook: eq.(6.2)

        Q[np.isnan(
            Q)] = 0  # All nan values should be put to zero, otherwise the corresponding frequency bin values will be nan forever
        # This also makes sense because Q=Qprev=0 means that in the next iteration Qprev will just not be taken into account
        # when calculating alpha

        Qprev = Q  # set previous value for next iteration

        noisevariance = np.vstack((noisevariance, Q))  # write in matrix

    ##the following for loops (for loops ftw ;)) are moving windows with length windowlength. They find the minimum psd per frequency bin
    # and replaces all values in the column with the minimum psd. a simplied version of this procedure is in the first half of my
    # testing.py function

    windowlength = int(1.5 / tsegment)  # segment in seconds to find minimum psd, respectively psd of noise
    numrows = noisevariance.shape[0]  # number of rows

    for rowstart, rowend in zip(range(0, numrows - windowlength, windowlength),
                                range(windowlength - 1, numrows, windowlength)):
        for k_column in range(0, noisevariance.shape[1]):
            noisevariance[list(range(rowstart, rowend + 1)), k_column] = min(
                noisevariance[list(range(rowstart, rowend + 1)), k_column])
            # Per Window (with length 'windowlength', which are number of rows):
            # Find the minimum per column and replace all the values in this column with the found minimum


    return noisevariance


def calculate_speechpsd_heuristic(F_data):
    # Calculate Speech PSD with the Heuristic Approach of Lect.1+4

    k = F_data.shape[1]  # number of freq bins
    R = F_data.shape[0]

    gainmatrix = np.empty(k)
    for j in range(0, R-1):
        fourier_row = F_data[j,:]  #take fourier of row
        psd_row=np.absolute(fourier_row)**2 #psd of row
        psd_row[psd_row == 0] = np.nan  # set nan to flag empty frequency bins
        speechpsd=psd_row-noisevariance[j,:] #simple formula from lect. 4

    return speechpsd

def calculate_wiener_gain(speechpsd,noisevariance):

    gain = speechpsd / (speechpsd + noisevariance[j, :])  # Wiener gain from lect.4
    gain[np.isnan(gain)] = 0 #set gain to zero for emmpty frequency bins, this means we will drop these values
    gainmatrix=np.vstack((gainmatrix,gain))  #stack in matrix

    return gainmatrix
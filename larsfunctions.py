import soundfile as sf
import numpy as np
import time #to use timer
import copy # to copy arrays, otherwise it will only be a reference


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


def calculate_noisepsd_min(F_data,tsegment,windowlength):
    #F_data is PSD (Bartlett Estimate!)

    ###
    ## MS Method (Hendricks Book eq.(6.2) + Martin2001 paper in Dropbox)
    ###
    ## TODO The Bias correction for alpha is missing!


    k = F_data.shape[1]  # number of freq bins
    Qprev = np.array(np.zeros(k))
    R = F_data.shape[0]
    #noisevariance=np.empty(k)
    noisevariance = np.empty([R,k])

    for j in range(0, R - 1):
        #fourier_row = F_data[j, :]  # load fourier of row
        #psd_row = np.absolute(fourier_row) ** 2  # psd of row
        psd_row = F_data[j,:]

        psd_row[psd_row == 0] = np.nan  # set nan to avoid division by zero
        alpha = 1 / (1 + (Qprev / psd_row - 1) ** 2)  # calculate alpha, see paper in dropbox
        #alpha = 0.85 #for testing

        onevector = np.array(np.ones(k))  # make onevector
        Q = alpha * Qprev + (onevector - alpha) * psd_row  # hendricksbook: eq.(6.2)

        Q[np.isnan(Q)] = 0  # All nan values should be put to zero, otherwise the corresponding frequency bin values will be nan forever
        # This also makes sense because Q=Qprev=0 means that in the next iteration Qprev will just not be taken into account
        # when calculating alpha

        Qprev = Q  # set previous value for next iteration
        #noisevariance = np.vstack((noisevariance, Q))  # write in matrix
        noisevariance[j,:] = Q


    ##the following for loops (for loops ftw ;)) are moving windows with length windowlength. They find the minimum psd per frequency bin
    # and replaces all values in the column with the minimum psd. a simplied version of this procedure is in the first half of my
    # testing.py function

    numcols=noisevariance.shape[1] #number of columns
    numrows = noisevariance.shape[0]  # number of rows

    noisevariance_minima=copy.copy(noisevariance)

    for rowstart, rowend in zip(range(0, numrows - windowlength, 1),range(windowlength - 1, numrows, 1)):
        noisevariance_minima[list(range(rowstart, rowend + 1)), 0:k+1] = np.amin(noisevariance[list(range(rowstart, rowend + 1)), :],axis=0)
            # Per Window (with length 'windowlength', which are number of rows):
            # Find the minimum per column and replace all the values in this column with the found minimum

    return noisevariance_minima


def calculate_speechpsd_heuristic(F_data,noisevariance):
    # Calculate Speech PSD with the Heuristic Approach of Lect.1+4

    k = F_data.shape[1]  # number of freq bins
    R = F_data.shape[0]

    speechpsdmatrix = np.empty(k)

    for j in range(0, R-1):
        fourier_row = F_data[j,:]  #take fourier of row
        psd_row=np.absolute(fourier_row)**2 #psd of row
        psd_row[psd_row == 0] = np.nan  # set nan to flag empty frequency bins
        speechpsd=psd_row-noisevariance[j,:] #simple formula from lect. 4
        speechpsdmatrix=np.vstack((speechpsdmatrix,speechpsd))  #stack in matrix

    return speechpsdmatrix


def calculate_wiener_gain(speechpsd,noisevariance):

    k = speechpsd.shape[1]
    R = speechpsd.shape[0]
    gainmatrix = np.empty(k)

    for j in range(0, R - 1):
        speechpsd[speechpsd == 0] = np.nan
        gain = speechpsd / (speechpsd + noisevariance[j, :])  # Wiener gain from lect.4
        gain[np.isnan(gain)] = 0 #set gain to zero for emmpty frequency bins, this means we will drop these values
        gainmatrix=np.vstack((gainmatrix,gain))  #stack in matrix


    return gainmatrix

def calculate_noisepsd_min_costas(data_seg_over,tsegment,windowlength):

    alpha=.5
    num_frames = data_seg_over.shape[0]
    k = data_seg_over.shape[1]  # number of freq bins
    Q = np.array(np.zeros(k))
    Qtot = np.array(np.zeros(k))

    for j in range(0, num_frames):
        Seg = np.fft.fft(data_seg_over[j, :])  # FFT each frame
        Seg[Seg == 0] = np.nan  # set nan to avoid division by zero
        alpha = 1 / (1 + (Q / Seg - 1) ** 2)
        psdY = np.absolute(Seg) ** 2  # periodogram of every frame
        Qnew = alpha*Q + (1-alpha)*psdY
        Qtot = np.vstack((Qtot, Qnew))
        Qnew[np.isnan(Qnew)] = 0
        Q = Qnew


    Qtot = np.delete(Qtot, (0), axis=0)


    L=100 # window from which the noise PSD is estimated [ for a 20msec frame -> L=2sec]
    psdN = np.zeros((100, 320))

    for i in range (L,num_frames):
        Qmin = Qtot[range(i-L, i)].min(0)
        psdN = np.vstack((psdN, Qmin))

    return psdN
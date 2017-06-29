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
    #data = np.pad(data, (0, int(remainder)), 'constant', constant_values=0)  # pad to subtract

    residual = data - reconstruction

    return residual



def calculate_noisepsd_min(F_data,tsegment,windowlength):
    #F_data is PSD (Bartlett Estimate!)

    ###
    ## MS Method (Hendricks Book eq.(6.2) + Martin2001 paper in Dropbox)
    ###
    ## TODO The Bias correction for alpha is missing!


    k = F_data.shape[1]  # number of freq bins
    R = F_data.shape[0]
    noisevariance = np.empty([R, k])
    noisevariance[0,:] = np.zeros(k)
    alphastack = np.empty([R, k])
    alphastack[0, :] = np.zeros(k)

    for j in range(1, R - 1):

        psd_row = copy.copy(F_data[j,:])
        psd_row_prev=copy.copy(F_data[j-1,:])

        psd_row[psd_row == 0] = np.nan  # set nan to avoid division by zero
        oldsigma=copy.copy(noisevariance[j-1,:])
        oldsigma[oldsigma == 0] = np.nan

        alpha = 1 / (1 + (psd_row_prev / oldsigma - 1) ** 2)  # calculate alpha, see paper in dropbox

        alpha[np.isnan(alpha)] = 0
        alpha[np.where(alpha >= 0.96)] = 0.96 #maximum alpha should be 0.96, see paper
        alpha[np.where(alpha < 0.3)] = 0.3

        #alpha = 0.85 #for testing

        onevector = np.array(np.ones(k))  # make onevector
        Q = alpha * noisevariance[j-1,:] + (onevector - alpha) * psd_row  # hendricksbook: eq.(6.2)

        Q[np.isnan(Q)] = 0  # All nan values should be put to zero, otherwise the corresponding frequency bin values will be nan forever
        # This also makes sense because Q=Qprev=0 means that in the next iteration Qprev will just not be taken into account
        # when calculating alpha

        alphastack[j,:]=alpha
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

    return noisevariance_minima,alphastack


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

def wiener(Py,Pn,y_k):
    Py[Py == 0] = np.nan
    gain= 1-Pn/Py
    gain[np.isnan(gain)] = 0
    s_est=gain*y_k

    return s_est

def Noise_MMSE(framed_data,fft_data,s_segment):
    num_frames = len(framed_data)
    Npsd = np.zeros([num_frames - 1, s_segment])

    k = 5  # number of frames from which the initial noise psd is estimated

    sigma_k = fft_data[0:k, :]
    #sigma_N = np.mean(np.absolute(sigma_k) ** 2, axis=0)  # averaging first k frames
    sigma_N = np.mean(sigma_k, axis=0)
    Npsd = np.vstack((sigma_N, Npsd))

    P_l = 0.5  # Initialize smoothened version of P(H1|y)
    a_PH1 = 0.9 #stagnation avoidance
    a_N = 0.8 #alphapow
    PH1 = 0.5  # Prior probability of speech presence
    PH0 = 1 - PH1  # Prior probability of speech absence
    ratio_P = PH1 / PH0
    ksi_H1_dB = 10  # Fixed a priori SNR
    ksi_H1 = 10 ** (ksi_H1_dB / 10)

    #signal_power = np.abs(framed_data) ** 2  # Dirty signal Power |y|^2
    signal_power=fft_data
    for j in range(0, num_frames):
        zeta = signal_power[j, :] / sigma_N  # a posteriori SNR
        PH1 = (1 + ratio_P * (1 + ksi_H1) * np.exp(- zeta * ksi_H1 / (1 + ksi_H1))) ** (-1)  # A posteriori SPP
        P_l = a_PH1 * P_l + (1 - a_PH1) * PH1  # Smoothen P(H1|y)
        PH1[P_l > 0.99] = min(PH1[P_l > 0.99], 0.99)
        MMSE_noise = PH0 * signal_power[j, :] + PH1 * sigma_N #eq. (22)
        sigma_N = a_N * sigma_N + (1 - a_N) * MMSE_noise #eq. (8)
        Npsd[j, :] = sigma_N

    return Npsd

def bartlett(psd_F_data):
    # Bartlett: split a time segment of 320 into smaller segments, take psd of smaller segments (make length 320 again)
    # and average over each of the segments
    # Step 2 of slide 76, lecture 1
    M=5
    k = psd_F_data.shape[1]  # number of freq bins
    R = psd_F_data.shape[0]
    numcols = copy.copy(k)  # number of columns
    numrows = copy.copy(R)  # number of rows
    bartlett_estimate = np.empty([R,k])

    bartlett_estimate[0:M-1,:] = np.zeros(k)

    for rowstart, rowend in zip(range(0, numrows - M, 1),range(M - 1, numrows, 1)):
        bartlett_estimate[rowend, 0:k+1] = np.mean(psd_F_data[list(range(rowstart, rowend + 1)), :],axis=0)
            # Per Window (with length 'windowlength', which are number of rows):
            # Find the minimum per column and replace all the values in this column with the found minimum

    return bartlett_estimate

def exponentialsmoother(psd_F_data,alpha):

    k = psd_F_data.shape[1]  # number of freq bins
    Qprev = np.array(np.zeros(k))
    R = psd_F_data.shape[0] #rows

    psd_F_data_smoothed = np.empty([R,k])
    psd_F_data_smoothed[0,:] = psd_F_data[0,:]

    for j in range(1, R - 1):
        onevector = np.array(np.ones(k))  # make onevector
        psd_F_data_smoothed[j,:] = alpha * psd_F_data_smoothed[j-1,:] + (onevector - alpha) * psd_F_data[j,:]  # hendricksbook: eq.(6.2)

    return psd_F_data_smoothed


def ml_estimation(bartlett_y,sigma_n):

    L = 2
    k = bartlett_y.shape[1]  # number of freq bins
    R = bartlett_y.shape[0] # number of frames
    numcols = copy.copy(k)  # number of columns
    numrows = copy.copy(R)  # number of rows

    sigma_s_ml = np.empty([R, k])

    for rowstart, rowend in zip(range(0, numrows - L, 1), range(L - 1, numrows, 1)):
        sigma_s_ml[rowend, 0:k + 1] = np.mean(bartlett_y[list(range(rowstart, rowend + 1)), :], axis=0) - sigma_n[rowend,0:k+1]

    return sigma_s_ml

def dd_approach(sigma_s,sigma_n,bartlett_y,alpha):

    k = bartlett_y.shape[1]  # number of freq bins
    R = bartlett_y.shape[0] # number of frames
    sigma_s_dd = np.empty([R, k])

    for j in range(1, R - 1):
        onevector = np.array(np.ones(k))  # make onevector
        sigma_s_dd[j,:] = alpha * sigma_s[j-1,:]/sigma_n[j,1] + (onevector - alpha) * np.maximum( (bartlett_y[j,:]/sigma_n[j,:])-1,0)  # hendricksbook: eq.(6.2)

    return sigma_s_dd
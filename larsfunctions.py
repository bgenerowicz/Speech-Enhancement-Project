def import_frame_data(filelocation):

    import soundfile as sf

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

    return rmsarray_han,samplerate




def transform_data(framed_data):

    
    F_data = np.fft.fft(rmsarray_han)

    return F_data
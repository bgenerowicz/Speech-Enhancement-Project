def readaudio(filelocation):
    import soundfile as sf

    newdata, samplerate = sf.read(filelocation)
    rms = [block for block in sf.blocks(filelocation, blocksize=320, overlap=160)]

    # Pad
    lastarr = len(rms) - 1  # take last array of the list
    length = len(rms[lastarr])  # calculate length of last array
    remainder = sseg - (length % sseg)  # calculate length of padding
    rms[lastarr] = np.pad(rms[lastarr], (0, int(remainder)), 'constant', constant_values=0)  # pad
    rmsarray = np.vstack(rms)

    # Create & Apply hanning window
    hanning_segment = np.hanning(rmsarray.shape[1])
    # rmsarray_han = np.multiply(hanning_segment,rmsarray)
    rmsarray_han = rmsarray  # for testing

    # FFT
    F_data = np.fft.fft(rmsarray_han)

    return rmsarray_han

import soundfile as sf

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
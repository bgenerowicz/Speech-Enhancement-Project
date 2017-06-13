import numpy as np

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
import numpy as np

def overlap_add(IF_data,len_data,s_segment,s_overlap):


    num_frames = int((len_data - s_overlap) / s_overlap)



    reconstructed_data = np.zeros(len_data)  # init matrix/ variables
    s_start = 0
    for i in range(0, num_frames):
        reconstructed_data[s_start:(s_start + s_segment)] = reconstructed_data[s_start:(s_start + s_segment)] + IF_data[
                                                                                                                i, :]
        s_start = s_start + s_overlap

    return reconstructed_data
import numpy as np


def Noise_MMSE(framed_data,s_segment,fft_data):
    num_frames = len(framed_data)
    Npsd = np.zeros([num_frames - 1, s_segment])

    k = 5  # number of frames from which the initial noise psd is estimated

    sigma_k = fft_data[0:k, :]
    sigma_N = np.mean(np.absolute(sigma_k) ** 2, axis=0)  # averaging first k frames
    Npsd = np.vstack((sigma_N, Npsd))

    P_l = 0.5  # Initialize smoothened version of P(H1|y)
    a_PH1 = 0.9
    a_N = 0.8
    PH1 = 0.5  # Prior probability of speech presence
    PH0 = 1 - PH1  # Prior probability of speech absence
    ratio_P = PH1 / PH0
    ksi_H1_dB = 10  # Fixed a priori SNR
    ksi_H1 = 10 ** (ksi_H1_dB / 10)

    signal_power = np.abs(framed_data) ** 2  # Dirty signal Power |y|^2

    for j in range(0, num_frames):
        zeta = signal_power[j, :] / sigma_N  # a posteriori SNR
        PH1 = (1 + ratio_P * (1 + ksi_H1) * np.exp(- zeta * ksi_H1 / (1 + ksi_H1))) ** (-1)  # A posteriori SPP
        P_l = a_PH1 * P_l + (1 - a_PH1) * PH1  # Smoothen P(H1|y)
        PH1[P_l > 0.99] = min(PH1[P_l > 0.99], 0.99)
        MMSE_noise = PH0 * signal_power[j, :] + PH1 * sigma_N
        sigma_N = a_N * sigma_N + (1 - a_N) * MMSE_noise
        Npsd[j, :] = sigma_N

    return Npsd

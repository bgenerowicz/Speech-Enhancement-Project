# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:09:04 2017

@author: sax
"""
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import math

import soundfile as sf

data, samplerate = sf.read('clean.wav')

plt.plot(data)

dataa = np.matrix(data)
F = np.zeros((3609, 320))
Seg_data = np.matrix(F)
frame = 320
l = len(data)
ovrlp = 0.5
Overfr = frame - round(frame * ovrlp)
NumFr = math.ceil((len(data) - frame) / (ovrlp * frame))
i = 0
for j in range(0, NumFr):
    Seg_data[j, 0:frame] = dataa[0, i:i + frame]
    i = i + Overfr
plt.isinteractive()

Rec1 = Seg_data[0, :]
Rec1.shape(1)
# {Rec_data = np.zeros((0,577690))
# for j in range(0, NumFr):
#     Rec_data[0, i:frame] = Seg_data[j, 0:frame / 2]

"""  MASK
one=np.ones(160)
zero=np.zeros(160)
mask=np.concatenate((one,zero))
"""

"""
for i in range(0,Numfr):
    Recon=Seg_data[:,j]

    numpy.delete(Seg_data, (161:320), axis=0)

    numpy.delete(, index)
    index=Seg_data[161:320] 
    """

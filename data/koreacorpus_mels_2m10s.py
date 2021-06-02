# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os
import sys
import numpy as np
sys.path.append('C:/nmb/nada/python_import/')
from feature_handling import load_data_mel

# female
# pathAudio_F = 'C:\\nmb\\nmb_data\\10s2m_korea_corpus\\korea_f_10s\\'
# load_data_mel(pathAudio_F, 'wav', 0)

# male
# pathAudio_M = 'C:\\nmb\\nmb_data\\10s2m_korea_corpus\\korea_m_10s\\'
# load_data_mel(pathAudio_M, 'wav', 1)

# file check
x1 = np.load('C:\\nmb\\nmb_data\\npy\\korea_corpus_2m10s\koreacorpus_2m10s_f_mel_data.npy')
print(x1.shape) # (228, 128, 1723)

y1 = np.load('C:\\nmb\\nmb_data\\npy\\korea_corpus_2m10s\koreacorpus_2m10s_f_mel_label.npy')
print(y1.shape) # (228,)
print(y1[:10])  # [0 0 0 0 0 0 0 0 0 0]

x2 = np.load('C:\\nmb\\nmb_data\\npy\\korea_corpus_2m10s\koreacorpus_2m10s_m_mel_data.npy')
print(x2.shape) # (240, 128, 1723)

y2 = np.load('C:\\nmb\\nmb_data\\npy\\korea_corpus_2m10s\koreacorpus_2m10s_m_mel_label.npy')
print(y2.shape) # (240,)
print(y2[:10])  # [1 1 1 1 1 1 1 1 1 1]

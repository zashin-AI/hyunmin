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
sys.path.append('E:/nmb/nada/python_import/')
from feature_handling import load_data_mel

# 정규화 (MinMaxScaler)

# female
# pathAudio_F = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\nslab_f\\f_total_chunk\\mindslab_f_total\\_noise\\'
# load_data_mel(pathAudio_F, 'wav', 0)

# male
# pathAudio_M = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\nslab_m\\m_total_chunk\\mindslab_m_total\\_noise\\'
# load_data_mel(pathAudio_M, 'wav', 1)


x1 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_denoise_1s2m_female_mel_data.npy')
print(x1.shape) # (12, 128, 862)

y1 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_denoise_1s2m_female_mel_label.npy')
print(y1.shape) # (12,)

x2 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_denoise_1s2m_male_mel_data.npy')
print(x2.shape) # (2, 128, 862)

y2 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_denoise_1s2m_male_mel_label.npy')
print(y2.shape) # (2,)

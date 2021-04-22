# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os
import sys
sys.path.append('E:/nmb/nada/python_import/')
from feature_handling import load_data_mel

# 정규화 (MinMaxScaler)

# female
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\korea_corpus_f_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\mindslab_f_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\open_slr_f_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\pansori_female_2m_noise\\'

# load_data_mel(pathAudio_M, 'wav', 0)

# male
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\korea_corpus_m_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\mindslab_m_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\open_slr_m_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\audio_data_noise\\pansori_male_2m_noise\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\test10\\'
# pathAudio_M = 'E:\\nmb\\nmb_data\\test30\\'
pathAudio_M = 'E:\\nmb\\nmb_data\\test1030\\'

load_data_mel(pathAudio_M, 'wav', 1)

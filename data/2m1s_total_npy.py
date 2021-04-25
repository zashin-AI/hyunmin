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

## data
f1 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\corpus_f_data.npy')
f2 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\mindslab_f_data.npy')
f3 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\pansori_f_data.npy')
f4 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\slr_f_data.npy')

m1 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\corpus_m_data.npy')
m2 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\mindslab_m_data.npy')
m3 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\pansori_m_data.npy')
m4 = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\slr_m_data.npy')

## label
f1_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\corpus_f_label.npy')
f2_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\mindslab_f_label.npy')
f3_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\pansori_f_label.npy')
f4_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\f\\slr_f_label.npy')

m1_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\corpus_m_label.npy')
m2_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\mindslab_m_label.npy')
m3_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\pansori_m_label.npy')
m4_l = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\m\\slr_m_label.npy')

## concatenate
x = np.concatenate([f1, f2, f3, f4, m1, m2, m3, m4], 0)
y = np.concatenate([f1_l, f2_l, f3_l, f4_l, m1_l, m2_l, m3_l, m4_l], 0)
print(x.shape)  # (19184, 128, 173)
print(y.shape)  # (19184,)

np.save('E:\\nmb\\nmb_data\\npy\\1m2s\\concate\\total_fm_data.npy', arr=x)
np.save('E:\\nmb\\nmb_data\\npy\\1m2s\\concate\\total_fm_label.npy', arr=y)

xx = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\concate\\total_fm_data.npy')
yy = np.load('E:\\nmb\\nmb_data\\npy\\1m2s\\concate\\total_fm_label.npy')
print(xx.shape, yy.shape)   # (19184, 128, 173) (19184,)
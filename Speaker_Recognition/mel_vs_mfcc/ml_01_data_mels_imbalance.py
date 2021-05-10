# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os

# 정규화 (MinMaxScaler)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

dataset = []
label = []
# 여 100 , 남 1200
# pathAudio_F = 'E:/nmb/nmb_data/imbalance_data/F100/F'
# pathAudio_M = 'E:/nmb/nmb_data/imbalance_data/F100/M/'

# 남 100, 여 1200
pathAudio_F = 'E:/nmb/nmb_data/imbalance_data/M100/F'
pathAudio_M = 'E:/nmb/nmb_data/imbalance_data/M100/M/'

files_F = librosa.util.find_files(pathAudio_F, ext=['flac'])
files_F_wav = librosa.util.find_files(pathAudio_F, ext=['wav'])
files_M = librosa.util.find_files(pathAudio_M, ext=['flac'])
files_M_wav = librosa.util.find_files(pathAudio_M, ext=['wav'])

files_F = np.array(files_F)
files_F_wav = np.array(files_F_wav)
files_F = np.append(files_F, files_F_wav)

files_M = np.asarray(files_M)
files_M_wav = np.asarray(files_M_wav)
files_M = np.append(files_M, files_M_wav)

print(files_F.shape)    # (100,)            ||| (1200,)
print(files_M.shape)    # (1200,)           ||| (100,)

total = [files_F, files_M]
index = 0               # index 0 : 여성, 1 : 남성

for folder in total : 
    print(f"===={index}=====")
    dataset = []
    label = []
    for file in folder:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
            mels = librosa.amplitude_to_db(mels, ref=np.max)

            dataset.append(mels)
            label.append(index)
    
    dataset = np.array(dataset)
    label = np.array(label)
    print(dataset.shape)    
    print(label.shape)      

    # np.save(f'E:/nmb/nmb_data/npy/imbalance_F100_{index}_mels.npy', arr=dataset)
    np.save(f'E:/nmb/nmb_data/npy/imbalance_M100_{index}_mels.npy', arr=dataset)
    print("dataset save")
    # np.save(f'E:/nmb/nmb_data/npy/imbalance_F100_{index}_mels_label.npy', arr=label)
    np.save(f'E:/nmb/nmb_data/npy/imbalance_M100_{index}_mels_label.npy', arr=label)
    print("label save")

    index += 1 


print('=====save done=====') 
# ------------------------------------------------------

# 여 100, 남 1200
# F_mels (label 0)
# (89, 128, 862)
# (89,)

# M_mels (label 1)
# (1037, 128, 862)
# (1037,)
# ***********************

# 여 1200, 남 100
# F_mels (label 0)
# (1104, 128, 862)
# (1104,)

# M_mels (label 1)
# (69, 128, 862)
# (69,)

# ------------------------------------------------------

# F = np.load('E:/nmb/nmb_data/npy/imbalance_F100_0_mels.npy')
F = np.load('E:/nmb/nmb_data/npy/imbalance_M100_0_mels.npy')
print(F.shape)  # (1104, 128, 862)
# M = np.load('E:/nmb/nmb_data/npy/imbalance_F100_1_mels.npy')
M = np.load('E:/nmb/nmb_data/npy/imbalance_M100_1_mels.npy')
print(M.shape)  # (1037, 128, 862)

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
pathAudio_F = 'E:/nmb/nmb_data/brandnew_dataset/F/'
pathAudio_M = 'E:/nmb/nmb_data/brandnew_dataset/M/'

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


print(files_F.shape)    # (1200,)
print(files_M.shape)    # (1200,)

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
            mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20, n_fft=512, hop_length=128)
            mfccs = normalize(mfccs, axis=1)

            # plt.figure(figsize=(10,4))
            # plt.title('MFCCs')
            # librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            # plt.colorbar()
            # plt.show()

            dataset.append(mfccs)
            label.append(index)
    
    dataset = np.array(dataset)
    label = np.array(label)
    print(dataset.shape)    
    print(label.shape)      

    np.save(f'E:/nmb/nmb_data/npy/brandnew_{index}_mfccs.npy', arr=dataset)
    print("dataset save")
    np.save(f'E:/nmb/nmb_data/npy/brandnew_{index}_mfccs_label.npy', arr=label)
    print("label save")

    index += 1 


print('=====save done=====') 
# ------------------------------------------------------

# F_mfccs (label 0)
# (1104, 20, 862)
# (1104,)

# M_mfccs (label 1)
# (1037, 20, 862)
# (1037,)

# ------------------------------------------------------

F = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs.npy')
print(F.shape)  # (1104, 20, 862)
M = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs.npy')
print(M.shape)  # (1037, 20, 862)

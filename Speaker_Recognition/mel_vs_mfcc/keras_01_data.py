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
pathAudio_M = 'E:/nmb/nmb_data/ForM/M/'
pathAudio_F = 'E:/nmb/nmb_data/ForM/F/'
files_M = librosa.util.find_files(pathAudio_M, ext=['flac'])
files_F = librosa.util.find_files(pathAudio_F, ext=['flac'])
files_M = np.asarray(files_M)
files_F = np.asarray(files_F)
print(files_M.shape)    # (557,)
print(files_F.shape)    # (557,)

total = [files_F, files_M]
index = 0

for folder in total : 
    dataset = []
    label = []
    for file in folder:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mfccs = librosa.feature.mfcc(y, sr=sr, hop_length=512, n_fft=512)
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

    np.save(f'E:/nmb/nmb_data/npy/pansori_{index}_mfccs.npy', arr=dataset)
    print("dataset save")
    np.save(f'E:/nmb/nmb_data/npy/pansori_{index}_label_mfccs.npy', arr=label)
    print("label save")

    index += 1 


print('=====save done=====') 
# ------------------------------------------------------

# F_mfccs (label 0)
# (545, 20, 216)
# (545,)

# M_mfccs (label 1)
# (528, 20, 216)
# (528,)

# ------------------------------------------------------

M = np.load('E:/nmb/nmb_data/npy/pansori_0_mfccs.npy')
print(M.shape)  # (545, 20, 216)
F = np.load('E:/nmb/nmb_data/npy/pansori_1_mfccs.npy')
print(F.shape)  # (528, 20, 216)

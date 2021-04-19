import numpy as np
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
import os

start_now = datetime.now()

# pred_pathAudio = 'E:/nmb/nmb_data/predict/ODD'
# files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
# files = np.asarray(files)
# count_odd = 0
# for file in files:   
#     name = os.path.basename(file)
#     name = name[0]

#     y, sr = librosa.load(file, sr=22050) 
#     mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
#     pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
#     pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
#     y_pred = model.predict(pred_mels)
#     y_pred_label = np.argmax(y_pred)
#     if y_pred_label == 0 :               
#         print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
#         if name == 'F' :
#             count_odd += 1
#     else:
#         count_m += 1                              
#         print(file, '{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))
#         if count_odd == 'M' :
            # count_odd += 1

pred = ['E:/nmb/nmb_data/predict/F','E:/nmb/nmb_data/predict/M','E:/nmb/nmb_data/predict/ODD']

for pred_pathAudio in pred : 

    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    count_f = 0
    count_m = 0
    count_odd = 0
    for file in files:   
        name = os.path.basename(file)
        # print(name)
        length = len(name)
        # print(length)
        name = name[0]

        y, sr = librosa.load(file, sr=22050) 
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0 :
            count_f += 1                 
            print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if name == 'F' :
                count_f = count_f + 1
            if length > 10 :    # 이상치
                if name == 'F' :
                    count_odd = count_odd + 1
        else:                         
            print(file, '{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if name == 'M' :
                count_m = count_f + 1
            if length > 10 :    # 이상치
                if name == 'M' :
                    count_odd = count_odd + 1


from itertools import count
import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler

start = datetime.now()

x = np.load('E:\\nmb\\nmb_data\\5s_last_0510\\total_data.npy')
y = np.load('E:\\nmb\\nmb_data\\5s_last_0510\\total_label.npy')

print(x.shape, y.shape) # (4536, 128, 862) (4536,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

x_train_shape_1 = x_train.shape[1]
x_train_shape_2 = x_train.shape[2]

x_test_shape_1 = x_test.shape[1]
x_test_shape_2 = x_test.shape[2]

x_train = x_train.reshape(x_train.shape[0], x_train_shape_1 * x_train_shape_2)
x_test = x_test.reshape(x_test.shape[0], x_test_shape_1 * x_test_shape_2)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train_shape_1 , x_train_shape_2)
x_test = x_test.reshape(x_test.shape[0], x_test_shape_1 , x_test_shape_2)

print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)

# 모델 구성

model = Sequential()

def residual_block(x, filters, conv_num=3, activation='relu'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Conv1D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = Add()([x,s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 23)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()


# 컴파일, 훈련
path = 'E:\\nmb\\nmb_data\\cp\\5s_deeplearning\\conv1d_2_batch16.h5'
op = Adam(lr=0.0001)
batch_size = 16

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model = load_model(path)
# model.load_weights('C:/nmb/nmb_data/h5/5s/Conv2D_1.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pred = ['E:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\F', 'E:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr=22050)
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0]* pred_mels.shape[1])

        pred_mels = scaler.transform(pred_mels)
        pred_mels = pred_mels.reshape(1, 128, 862)

        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:   # 여성이라고 예측
            print(file, '{:.4f} 의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if name == 'F' :
                count_f += 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f} 의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if name == 'M' :
                count_m += 1
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start
print("작업 시간 : ", time)

import winsound as sd
def beepsound():
    fr = 440    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

"""
Epoch 00024: early stopping
57/57 [==============================] - 1s 19ms/step - loss: 0.0751 - acc: 0.9692
loss : 0.07508
acc : 0.96916
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F1.wav 99.9972 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F10.wav 99.9350 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F11.wav 99.9886 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F12.wav 99.9999 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F13.wav 99.9873 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F14.wav 99.9997 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F15.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F16.wav 99.9848 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F17.wav 99.9997 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F18.wav 99.9968 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F19.wav 99.9145 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F2.wav 99.9995 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F20.wav 99.9998 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F21.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F22.wav 97.7340 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F23.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F24.wav 99.9980 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F25.wav 99.9997 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F26.wav 99.9913 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F27.wav 99.9995 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F28.wav 99.9998 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F29.wav 99.2428 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F3.wav 99.9990 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F30.wav 99.9830 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F31.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F32.wav 96.7448 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F33.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F34.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F35.wav 99.6825 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F36.wav 99.9741 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F37.wav 99.8791 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F38.wav 99.9965 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F39.wav 99.9948 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F4.wav 99.9973 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F40.wav 99.9994 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F41.wav 99.9826 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F42.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F43.wav 99.9906 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F5.wav 99.9994 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F6.wav 99.9846 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F7.wav 86.4705 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F8.wav 99.9982 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F9.wav 99.9985 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M1.wav 99.9994 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M10.wav 99.9996 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M11.wav 99.9920 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M12.wav 99.9800 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M13.wav 90.4011 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M14.wav 99.9521 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M15.wav 99.9991 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M16.wav 99.7288 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M17.wav 99.6343 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M18.wav 99.9996 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M19.wav 99.9983 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M2.wav 99.9395 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M20.wav 99.9998 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M21.wav 99.9928 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M22.wav 99.9142 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M23.wav 99.9999 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M24.wav 99.7924 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M25.wav 79.7355 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M26.wav 99.9739 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M27.wav 99.9825 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M28.wav 99.8549 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M29.wav 99.8549 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M3.wav 99.9874 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M30.wav 76.3500 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M31.wav 99.9996 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M32.wav 99.9953 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M33.wav 99.0390 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M34.wav 99.9963 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M35.wav 99.9927 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M36.wav 99.8526 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M37.wav 99.9968 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M38.wav 99.9996 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M39.wav 99.9964 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M4.wav 99.9985 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M40.wav 99.9998 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M41.wav 91.1668 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M42.wav 99.9837 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M43.wav 93.5045 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M5.wav 99.6823 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M6.wav 99.9999 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M7.wav 67.7087 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M8.wav 99.9981 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M9.wav 99.9720 의 확률로 남자입니다.
43개 여성 목소리 중 42개 정답
42개 남성 목소리 중 41개 정답
작업 시간 :  0:02:23.364286
"""

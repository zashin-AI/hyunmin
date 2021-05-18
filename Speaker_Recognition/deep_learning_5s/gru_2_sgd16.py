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
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop, SGD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.layers.wrappers import Bidirectional
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

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628, 2)
# print(x_test.shape, y_test.shape)   # (908, 128, 862) (908, 2)

# 모델 구성

model = Sequential()

def residual_block(x, units, conv_num=3, activation='tanh'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = GRU(units, return_sequences=True)(x) 
    for i in range(conv_num - 1):
        x = GRU(units, return_sequences=True)(x) # return_sequences=True 이거 사용해서 lstm shape 부분 3차원으로 맞춰줌 -> 자세한 내용 찾아봐야함
        x = Activation(activation)(x)
    x = GRU(units)(x)
    x = Add()([x,s])
    return Activation(activation)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 1024, 2)
    x = residual_block(x, 512, 2)
    x = residual_block(x, 512, 3)
    x = residual_block(x, 256, 3)
    x = residual_block(x, 256, 3)

    x = Bidirectional(GRU(16))(x)  #  LSTM 레이어 부분에 Bidirectional() 함수 -> many to one 유형
    x = Dense(256, activation="tanh")(x)
    x = Dense(128, activation="tanh")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# 컴파일, 훈련
path = 'E:\\nmb\\nmb_data\\cp\\5s_deeplearning\\gru_2_sgd16.h5'
op = SGD(lr=1e-3)
batch_size = 16

es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.4, patience=20, verbose=1)
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
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

"""
Epoch 00180: early stopping
57/57 [==============================] - 8s 89ms/step - loss: 0.3553 - acc: 0.8711
loss : 0.35526
acc : 0.87115
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F1.wav 96.7607 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F10.wav 97.3683 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F11.wav 98.5756 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F12.wav 98.4719 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F13.wav 99.2566 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F14.wav 92.7674 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F15.wav 89.8566 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F16.wav 88.8664 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F17.wav 90.8576 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F18.wav 99.0164 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F19.wav 97.6463 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F2.wav 77.9777 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F20.wav 90.1598 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F21.wav 92.0192 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F22.wav 96.6266 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F23.wav 67.9463 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F24.wav 94.8987 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F25.wav 98.9697 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F26.wav 97.0816 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F27.wav 97.4815 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F28.wav 78.9784 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F29.wav 97.6225 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F3.wav 67.7407 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F30.wav 83.9916 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F31.wav 83.3370 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F32.wav 82.6740 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F33.wav 99.9772 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F34.wav 99.9772 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F35.wav 96.7541 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F36.wav 97.1408 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F37.wav 97.2260 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F38.wav 70.3407 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F39.wav 98.5470 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F4.wav 86.9542 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F40.wav 75.2906 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F41.wav 99.9163 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F42.wav 82.1120 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F43.wav 68.8805 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F5.wav 67.0319 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F6.wav 58.8339 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F7.wav 91.6747 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F8.wav 99.3648 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F9.wav 88.2203 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M1.wav 93.5746 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M10.wav 92.6710 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M11.wav 92.5956 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M12.wav 91.0225 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M13.wav 98.3287 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M14.wav 87.1091 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M15.wav 93.8920 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M16.wav 94.8104 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M17.wav 90.5150 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M18.wav 95.5063 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M19.wav 99.9064 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M2.wav 74.8646 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M20.wav 77.7507 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M21.wav 98.5379 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M22.wav 86.0421 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M23.wav 71.0751 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M24.wav 58.0598 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M25.wav 93.0351 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M26.wav 70.5477 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M27.wav 92.8369 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M28.wav 79.2702 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M29.wav 79.2702 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M3.wav 63.4432 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M30.wav 90.4803 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M31.wav 96.0514 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M32.wav 96.5734 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M33.wav 99.9055 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M34.wav 97.2191 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M35.wav 54.5310 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M36.wav 72.9798 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M37.wav 82.1440 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M38.wav 93.2486 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M39.wav 91.4593 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M4.wav 89.0777 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M40.wav 95.4709 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M41.wav 52.0749 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M42.wav 87.4814 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M43.wav 99.3369 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M5.wav 51.2495 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M6.wav 69.1414 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M7.wav 73.9240 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M8.wav 65.2803 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M9.wav 96.9245 의 확률로 남자입니다.
43개 여성 목소리 중 39개 정답
42개 남성 목소리 중 30개 정답
작업 시간 :  2:16:33.092604
"""
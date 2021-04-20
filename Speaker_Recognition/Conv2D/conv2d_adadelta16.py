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

start = datetime.now()

# 데이터 불러오기
f_ds = np.load('E:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_f_mels.npy')
f_lb = np.load('E:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_f_label_mels.npy')
m_ds = np.load('E:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_mels.npy')
m_lb = np.load('E:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_label_mels.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) 
# (3840, 128, 862) (3840,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42
)
aaa = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3072, 128, 862, 1) (3072,)
print(x_test.shape, y_test.shape)   # (768, 128, 862, 1) (768,) 

# 모델 구성
model = Sequential()
def residual_block(x, filters, conv_num=3, activation='relu'): 
    # Shortcut
    s = Conv2D(filters, 1, padding='same')(x)

    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    # x = residual_block(x, 128, 3)
    # x = residual_block(x, 128, 3)

    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)
model.summary()

# model.save('E:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta.h5')

# 컴파일, 훈련
op = Adadelta(lr=0.001)
batch_size = 16

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
# path = 'E:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta001_16.h5'
path = 'E:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta0001_16.h5'
# path = 'E:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta00001_16.h5'
# path = 'E:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta000001_16.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='E:\\nmb\\nmb_data\\graph\\'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])

# 평가, 예측
model.load_weights(path)
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

pred = ['E:/nmb/nmb_data/predict/F','E:/nmb/nmb_data/predict/M','E:/nmb/nmb_data/predict/ODD']

count_f = 0
count_m = 0
count_odd = 0

for pred_pathAudio in pred : 
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:   
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr=22050) 
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0 :  # 여성이라고 예측
            print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if length > 9 :    # 이상치
                if name == 'F' :
                    count_odd = count_odd + 1                   
            else :
                if name == 'F' :
                    count_f = count_f + 1
                
        else:                   # 남성이라고 예측              
            print(file,'{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if length > 9 :    # 이상치
                if name == 'M' :
                    count_odd = count_odd + 1
            else :
                if name == 'M' :
                    count_m = count_m + 1
                
                    
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")
print("10개 이상치 목소리 중 "+str(count_odd)+"개 정답")


end = datetime.now()
time = end - start
print("작업 시간 : " , time)  

import winsound as sd
def beepsound():
    fr = 800    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input (InputLayer)              [(None, 128, 862, 1) 0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 862, 16) 160         input[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 862, 16) 0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 862, 16) 2320        activation[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 862, 16) 32          input[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 128, 862, 16) 0           conv2d_2[0][0]
                                                                 conv2d[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 862, 16) 0           add[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 127, 861, 16) 0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 127, 861, 32) 4640        max_pooling2d[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 127, 861, 32) 0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 127, 861, 32) 9248        activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 127, 861, 32) 544         max_pooling2d[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 127, 861, 32) 0           conv2d_5[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 127, 861, 32) 0           add_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 126, 860, 32) 0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 126, 860, 64) 18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 126, 860, 64) 0           conv2d_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 126, 860, 64) 36928       activation_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 126, 860, 64) 0           conv2d_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 126, 860, 64) 36928       activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 126, 860, 64) 2112        max_pooling2d_1[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 126, 860, 64) 0           conv2d_9[0][0]
                                                                 conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 126, 860, 64) 0           add_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 125, 859, 64) 0           activation_6[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 41, 286, 64)  0           max_pooling2d_2[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 750464)       0           average_pooling2d[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          192119040   flatten[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 128)          32896       dense[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            258         dense_1[0][0]
==================================================================================================
Total params: 192,263,602
Trainable params: 192,263,602
Non-trainable params: 0
__________________________________________________________________________________________________
'''
'''
loss : 0.30972
acc : 0.85807
E:\nmb\nmb_data\predict\F\F1.wav 94.8618 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F10.wav 84.1217 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F11.wav 58.2280 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F12.wav 91.2127 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F13.wav 89.4805 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F14.wav 95.8109 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F15.wav 96.2498 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F16.wav 81.7972 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F17.wav 95.5590 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F18.wav 96.2703 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F19.wav 96.6060 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F2.wav 98.2439 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F20.wav 97.9736 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F21.wav 98.3217 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F22.wav 95.7356 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F23.wav 93.5660 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F24.wav 95.7387 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F25.wav 95.9595 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F26.wav 97.5854 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F27.wav 92.5305 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F28.wav 86.3099 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F29.wav 98.4025 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F3.wav 74.8504 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F30.wav 91.8278 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F31.wav 99.8617 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F32.wav 95.6673 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F33.wav 88.3842 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F34.wav 97.9617 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F35.wav 87.3921 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F36.wav 98.0329 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F37.wav 97.9915 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F38.wav 99.5625 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F39.wav 93.2908 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F4.wav 97.4216 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F40.wav 90.1173 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F41.wav 73.0211 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F42.wav 95.8503 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F43.wav 94.7515 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F5.wav 98.6583 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F6.wav 96.2858 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F7.wav 85.9431 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F8.wav 83.6157 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F9.wav 61.0068 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M1.wav 61.9172 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M10.wav 87.7920 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M11.wav 99.1527 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M12.wav 98.1262 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M13.wav 99.7399 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M14.wav 70.9064 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M15.wav 86.0468 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M16.wav 96.4580 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M17.wav 79.9967 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M18.wav 98.9928 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M19.wav 99.2759 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M2.wav 69.1246 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M20.wav 95.7257 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M21.wav 72.1547 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M22.wav 99.9527 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M23.wav 94.8835 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M24.wav 51.4645 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M25.wav 75.2881 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M26.wav 97.7208 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M27.wav 90.6473 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M28.wav 96.5479 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M29.wav 78.7407 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M3.wav 93.5044 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M30.wav 81.3183 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M31.wav 56.5109 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M32.wav 83.3139 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M33.wav 95.8383 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M34.wav 94.6692 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M35.wav 80.2943 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M36.wav 51.5760 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M37.wav 94.4028 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M38.wav 86.9007 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M39.wav 91.3916 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M4.wav 99.2771 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M40.wav 96.8708 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M41.wav 96.6616 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M42.wav 95.4575 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M43.wav 52.6061 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M5.wav 94.3539 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M6.wav 99.6471 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M8.wav 99.3503 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M9.wav 99.6989 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F1_high.wav 51.3263 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\F2_high.wav 68.7405 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\F2_low.wav 64.8446 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F3_high.wav 90.9731 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M2_high.wav 86.1700 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M2_low.wav 97.7374 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M5_high.wav 79.5253 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M5_low.wav 95.4934 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M7_high.wav 51.4588 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M7_low.wav 99.9806 %의 확률로 남자입니다.
43개 여성 목소리 중 42개 정답
42개 남성 목소리 중 35개 정답
10개 이상치 목소리 중 8개 정답
작업 시간 :  9:09:39.190391



loss : 0.14335
acc : 0.93750
E:\nmb\nmb_data\predict\F\F1.wav 99.5568 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F10.wav 92.9466 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F11.wav 93.4553 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F12.wav 99.5464 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F13.wav 97.3536 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F14.wav 99.9598 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F15.wav 99.8111 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F16.wav 89.6537 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F17.wav 99.3333 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F18.wav 99.8558 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F19.wav 99.2340 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F2.wav 99.1221 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F20.wav 99.6241 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F21.wav 99.9223 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F22.wav 97.0376 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F23.wav 99.8295 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F24.wav 99.6042 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F25.wav 99.2042 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F26.wav 99.5077 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F27.wav 99.8019 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F28.wav 99.4568 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F29.wav 99.8841 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F3.wav 97.4741 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F30.wav 88.8423 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F31.wav 99.9795 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F32.wav 99.6662 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F33.wav 98.4390 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F34.wav 99.8942 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F35.wav 99.1328 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F36.wav 98.6450 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F37.wav 99.1946 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F38.wav 99.9558 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F39.wav 99.2238 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F4.wav 99.8937 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F40.wav 92.4865 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F41.wav 89.8771 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F42.wav 99.9327 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F43.wav 96.1829 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F5.wav 99.8338 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F6.wav 99.0124 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F7.wav 75.0088 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F8.wav 58.4379 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F9.wav 96.5629 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M1.wav 99.8607 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M10.wav 98.9133 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M11.wav 99.8914 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M12.wav 99.7916 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M13.wav 99.9877 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M14.wav 55.9975 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M15.wav 99.0098 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M16.wav 88.3030 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M17.wav 82.6464 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M18.wav 99.8084 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M19.wav 99.8578 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M2.wav 99.0129 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M20.wav 99.8894 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M21.wav 99.3715 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M22.wav 99.9916 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M23.wav 99.1578 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M24.wav 93.7298 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M25.wav 99.8062 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M26.wav 99.6687 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M27.wav 97.6017 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M28.wav 99.5582 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M29.wav 97.0806 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M3.wav 99.9333 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M30.wav 99.4298 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M31.wav 88.0734 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M32.wav 99.8866 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M33.wav 99.9816 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M34.wav 97.7093 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M35.wav 98.5945 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M36.wav 75.9465 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M37.wav 99.8501 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M38.wav 99.5421 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M39.wav 99.4031 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M4.wav 99.9536 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M40.wav 99.9784 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M41.wav 97.3830 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M42.wav 99.9293 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M43.wav 79.8823 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M5.wav 94.4240 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M6.wav 99.9977 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M8.wav 97.4875 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M9.wav 99.9994 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F1_high.wav 70.5044 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\F2_high.wav 63.8810 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F2_low.wav 93.5243 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F3_high.wav 89.5675 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M2_high.wav 80.9909 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M2_low.wav 99.9824 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M5_high.wav 57.6021 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\M5_low.wav 99.8344 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M7_high.wav 77.6939 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\M7_low.wav 99.9981 %의 확률로 남자입니다.
43개 여성 목소리 중 41개 정답
42개 남성 목소리 중 39개 정답
10개 이상치 목소리 중 5개 정답
작업 시간 :  3:34:38.405780

lr 0.001
loss : 0.12140
acc : 0.95833
E:\nmb\nmb_data\predict\F\F1.wav 99.9904 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F10.wav 99.1847 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F11.wav 98.4940 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F12.wav 99.8621 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F13.wav 99.9248 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F14.wav 99.9989 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F15.wav 99.9833 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F16.wav 90.2098 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F17.wav 99.9824 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F18.wav 99.9936 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F19.wav 99.9788 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F2.wav 99.9927 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F20.wav 99.9922 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F21.wav 99.9950 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F22.wav 99.6950 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F23.wav 99.9922 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F24.wav 99.9648 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F25.wav 99.9603 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F26.wav 99.9795 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F27.wav 99.9955 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F28.wav 99.9827 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F29.wav 99.9940 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F3.wav 99.7730 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F30.wav 98.3512 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F31.wav 99.9996 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F32.wav 99.9941 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F33.wav 99.9375 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F34.wav 99.9970 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F35.wav 99.9707 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F36.wav 99.8522 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F37.wav 99.9823 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F38.wav 99.9988 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F39.wav 99.9675 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F4.wav 99.9957 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F40.wav 99.8893 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F41.wav 83.8495 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F42.wav 99.9965 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F43.wav 99.8999 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F5.wav 99.9830 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F6.wav 99.9940 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\F\F7.wav 93.5100 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F8.wav 81.5476 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\F\F9.wav 99.6567 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M1.wav 99.9964 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M10.wav 99.9971 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M11.wav 99.9943 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M12.wav 99.9954 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M13.wav 99.9997 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M14.wav 83.2134 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M15.wav 99.8281 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M16.wav 55.4630 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\M\M17.wav 72.3372 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M18.wav 99.9996 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M19.wav 99.9996 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M2.wav 99.9919 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M20.wav 99.9997 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M21.wav 99.9822 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M22.wav 99.9997 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M23.wav 99.8637 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M24.wav 96.2045 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M25.wav 99.9994 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M26.wav 99.9981 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M27.wav 99.8829 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M28.wav 99.9954 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M29.wav 99.8947 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M3.wav 99.9996 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M30.wav 99.9993 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M31.wav 99.3414 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M32.wav 99.9918 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M33.wav 99.9996 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M34.wav 99.8469 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M35.wav 99.4111 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M36.wav 77.3042 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M37.wav 99.9979 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M38.wav 99.9983 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M39.wav 99.9136 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M4.wav 99.9952 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M40.wav 99.9997 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M41.wav 99.7792 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M42.wav 99.9994 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M43.wav 99.3545 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M5.wav 99.9559 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M6.wav 99.9998 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M8.wav 92.3931 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\M\M9.wav 100.0000 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F1_high.wav 63.0041 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F2_high.wav 98.2471 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F2_low.wav 97.6885 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\F3_high.wav 57.2576 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\M2_high.wav 96.4083 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M2_low.wav 100.0000 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M5_high.wav 94.7967 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M5_low.wav 99.9996 %의 확률로 남자입니다.
E:\nmb\nmb_data\predict\ODD\M7_high.wav 57.2983 %의 확률로 여자입니다.
E:\nmb\nmb_data\predict\ODD\M7_low.wav 100.0000 %의 확률로 남자입니다.
43개 여성 목소리 중 39개 정답
42개 남성 목소리 중 41개 정답
10개 이상치 목소리 중 6개 정답
작업 시간 :  2:10:51.926545
'''
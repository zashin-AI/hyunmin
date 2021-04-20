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
f_ds = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_f_mels.npy')
f_lb = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_f_label_mels.npy')
m_ds = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_mels.npy')
m_lb = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_label_mels.npy')

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

# model = load_model('C:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta000001_8.h5')

# 컴파일, 훈련
op = Adadelta(lr=0.01)
batch_size = 8

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta001_8.h5'
# path = 'C:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta0001_8.h5'
# path = 'C:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta00001_8.h5'
# path = 'C:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta000001_8.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:\\nmb\\nmb_data\\graph\\'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])

# 평가, 예측
model.load_weights(path)
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

pred = ['C:/nmb/nmb_data/predict/F','C:/nmb/nmb_data/predict/M','C:/nmb/nmb_data/predict/ODD']

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
loss : 0.32508
acc : 0.85938
C:\nmb\nmb_data\predict\F\F1.wav 73.1202 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F10.wav 92.6299 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F11.wav 80.0961 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F12.wav 92.9249 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F13.wav 78.6880 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F14.wav 94.9956 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F15.wav 95.0090 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F16.wav 81.6232 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F17.wav 95.7071 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F18.wav 88.0685 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F19.wav 92.1470 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F2.wav 85.0010 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F20.wav 92.5375 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F21.wav 96.9836 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F22.wav 52.1951 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F23.wav 91.2834 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F24.wav 98.2650 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F25.wav 95.8876 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F26.wav 95.2559 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F27.wav 89.2221 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F28.wav 79.4395 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F29.wav 99.6174 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F3.wav 96.1202 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F30.wav 88.3021 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F31.wav 99.1671 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F32.wav 89.5484 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F33.wav 74.6117 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F34.wav 95.1862 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F35.wav 96.1500 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F36.wav 94.6815 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F37.wav 97.2665 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F38.wav 99.5537 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F39.wav 95.4220 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F4.wav 92.9815 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F40.wav 74.4571 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F41.wav 60.9132 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F42.wav 94.2011 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F43.wav 67.0980 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F5.wav 99.7491 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F6.wav 93.3517 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F7.wav 81.8543 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F8.wav 67.5248 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F9.wav 55.4332 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M1.wav 79.7545 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M10.wav 62.1979 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M11.wav 97.9541 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M12.wav 99.1773 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M13.wav 99.7188 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M14.wav 73.9736 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M15.wav 95.6602 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M16.wav 95.1523 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M17.wav 90.8603 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M18.wav 99.8403 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M19.wav 98.7894 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M2.wav 65.4100 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M20.wav 99.1424 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M21.wav 94.8457 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M22.wav 99.9949 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M23.wav 87.2394 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M24.wav 65.4270 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M25.wav 86.8988 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M26.wav 98.6137 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M27.wav 92.7898 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M28.wav 97.8425 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M29.wav 95.7226 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M3.wav 75.3214 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M30.wav 92.9415 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M31.wav 66.6765 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M32.wav 93.7524 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M33.wav 99.9876 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M34.wav 84.6164 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M35.wav 72.4517 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M36.wav 64.9278 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M37.wav 93.3425 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M38.wav 62.1210 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M39.wav 96.5917 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M4.wav 97.9271 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M40.wav 91.6881 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M41.wav 91.3420 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M42.wav 98.5952 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M43.wav 64.2437 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M5.wav 95.7912 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M6.wav 99.8062 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M8.wav 96.9524 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M9.wav 99.8779 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F1_high.wav 77.5108 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\F2_high.wav 74.6384 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\F2_low.wav 92.4945 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F3_high.wav 91.9426 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_high.wav 90.1534 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_low.wav 99.4040 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_high.wav 55.1658 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\M5_low.wav 97.3883 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_high.wav 53.0189 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\M7_low.wav 99.9939 %의 확률로 남자입니다.
43개 여성 목소리 중 39개 정답
42개 남성 목소리 중 35개 정답
10개 이상치 목소리 중 6개 정답
작업 시간 :  3:14:43.456467


lr 0.001 
loss : 0.12893
acc : 0.95573
C:\nmb\nmb_data\predict\F\F1.wav 99.9745 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F10.wav 98.7687 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F11.wav 98.6269 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F12.wav 99.9505 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F13.wav 99.8729 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F14.wav 99.9998 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F15.wav 99.9953 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F16.wav 94.5193 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F17.wav 99.9928 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F18.wav 99.9978 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F19.wav 99.9904 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F2.wav 99.9937 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F20.wav 99.9974 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F21.wav 99.9984 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F22.wav 99.9039 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F23.wav 99.9977 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F24.wav 99.9839 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F25.wav 99.9702 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F26.wav 99.9872 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F27.wav 99.9964 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F28.wav 99.9795 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F29.wav 99.9976 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F3.wav 99.6237 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F30.wav 97.9407 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F31.wav 99.9999 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F32.wav 99.9981 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F33.wav 99.8471 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F34.wav 99.9985 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F35.wav 99.9782 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F36.wav 99.8367 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F37.wav 99.9850 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F38.wav 99.9998 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F39.wav 99.9817 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F4.wav 99.9948 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F40.wav 99.8656 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F41.wav 58.9031 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F42.wav 99.9977 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F43.wav 99.7911 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F5.wav 99.9782 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F6.wav 99.9975 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F7.wav 90.2760 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F8.wav 76.7793 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F9.wav 99.8633 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M1.wav 99.9993 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M10.wav 99.9984 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M11.wav 99.9986 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M12.wav 99.9968 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M13.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M14.wav 96.6256 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M15.wav 99.5707 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M16.wav 76.7904 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M17.wav 77.4631 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M18.wav 99.9998 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M19.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M2.wav 99.9923 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M20.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M21.wav 99.9972 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M22.wav 99.9998 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M23.wav 99.9404 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M24.wav 98.1457 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M25.wav 99.9998 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M26.wav 99.9982 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M27.wav 99.9111 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M28.wav 99.9960 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M29.wav 99.7171 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M3.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M30.wav 99.9985 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M31.wav 99.1784 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M32.wav 99.9945 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M33.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M34.wav 99.9678 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M35.wav 99.5405 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M36.wav 92.0086 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M37.wav 99.9996 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M38.wav 99.9982 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M39.wav 99.8819 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M4.wav 99.9995 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M40.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M41.wav 99.9056 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M42.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M43.wav 98.0457 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M5.wav 99.9692 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M6.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M8.wav 94.5341 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M9.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F1_high.wav 66.0383 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F2_high.wav 98.3154 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F2_low.wav 96.1910 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F3_high.wav 69.5100 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_high.wav 82.4694 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_low.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_high.wav 93.1770 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_low.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_high.wav 88.8295 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\M7_low.wav 100.0000 %의 확률로 남자입니다.
43개 여성 목소리 중 39개 정답
42개 남성 목소리 중 41개 정답
10개 이상치 목소리 중 5개 정답
작업 시간 :  2:05:53.050638

lr=0.01
loss : 0.13163
acc : 0.95833
C:\nmb\nmb_data\predict\F\F1.wav 99.9995 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F10.wav 99.5928 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F11.wav 99.8857 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F12.wav 99.9921 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F13.wav 99.9899 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F14.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F15.wav 99.9998 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F16.wav 81.9242 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F17.wav 99.9951 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F18.wav 99.9999 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F19.wav 99.9996 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F2.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F20.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F21.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F22.wav 99.9516 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F23.wav 99.9997 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F24.wav 99.9901 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F25.wav 99.9469 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F26.wav 99.9687 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F27.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F28.wav 99.9982 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F29.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F3.wav 99.9988 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F30.wav 98.8896 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F31.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F32.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F33.wav 99.9567 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F34.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F35.wav 99.9999 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F36.wav 99.9300 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F37.wav 99.9999 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F38.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F39.wav 99.9996 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F4.wav 99.9997 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F40.wav 99.9990 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F41.wav 93.2722 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F42.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F43.wav 99.9975 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F5.wav 98.5480 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F6.wav 100.0000 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F7.wav 92.5472 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F\F8.wav 96.1735 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F\F9.wav 99.9951 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M1.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M10.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M11.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M12.wav 99.9997 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M13.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M14.wav 88.0436 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M15.wav 99.9762 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M16.wav 99.5606 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M17.wav 96.9679 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M18.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M19.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M2.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M20.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M21.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M22.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M23.wav 99.8507 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M24.wav 99.9936 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M25.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M26.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M27.wav 99.9983 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M28.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M29.wav 99.9995 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M3.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M30.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M31.wav 99.9599 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M32.wav 99.9999 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M33.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M34.wav 99.9987 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M35.wav 99.3211 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M36.wav 82.7033 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M37.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M38.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M39.wav 99.9993 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M4.wav 99.9998 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M40.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M41.wav 99.9789 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M42.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M43.wav 99.8636 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M5.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M6.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M\M8.wav 61.3436 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M\M9.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F1_high.wav 97.4666 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\F2_high.wav 87.4716 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F2_low.wav 99.9802 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\F3_high.wav 99.7158 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\M2_high.wav 80.4473 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\ODD\M2_low.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_high.wav 87.8794 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_low.wav 100.0000 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_high.wav 68.6225 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_low.wav 100.0000 %의 확률로 남자입니다.
43개 여성 목소리 중 41개 정답
42개 남성 목소리 중 40개 정답
10개 이상치 목소리 중 7개 정답
작업 시간 :  0:33:44.704093
'''
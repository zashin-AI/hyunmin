# https://keras.io/examples/audio/speaker_recognition_using_cnn/
# 참고해서 모델 만들기

# MFCCs 를 거친 데이터를 인풋으로!!!

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# 데이터 불러오기
f_ds = np.load('E:/nmb/nmb_data/npy/pansori_0_mfccs.npy')
f_lb = np.load('E:/nmb/nmb_data/npy/pansori_0_label_mfccs.npy')
m_ds = np.load('E:/nmb/nmb_data/npy/pansori_1_mfccs.npy')
m_lb = np.load('E:/nmb/nmb_data/npy/pansori_1_label_mfccs.npy')


x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (1073, 20, 216)
print(y.shape)  # (1073,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.15, random_state=42)
print(x_train.shape)    # (912, 20, 216)
print(x_test.shape)     # (161, 20, 216)
print(y_train.shape)    # (912,)
print(y_test.shape)     # (161,)


# 모델 구성
model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])                                           # add : for문을 통과한 x 가중치 & 맨 위에 있는 s 가중치를 합친다.
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)


def build_model(input_shape, num_classes):                      # (input shape, outut dense node)
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3)(x)             # AveragePolling : 평균함
    x = Flatten()(x)                                            # fully connected layer : Flatten을 사용해서 Conv1D와 Dense를 이어준다.
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)    # softmax : 결국 우리가 구분하고자 하는 화자는 여러 명이기 때문에 softmax를 사용함

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)       
# print(x_train.shape[1:])    # (20, 216)


model.summary()

# 컴파일, 훈련
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
mcpath = 'E:/nmb/nmb_data/cp/conv1_model_01_mfccs.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.1, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights('E:/nmb/nmb_data/cp/conv1_model_01_mfccs.h5')

result = model.evaluate(x_test, y_test, batch_size=16)
print('loss: ', result[0])
print('acc: ', result[1])

pred_pathAudio = 'E:/nmb/nmb_data/teamvoice/clear/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mfccs = librosa.feature.mfcc(y, sr=sr, hop_length=512, n_fft=512)
    pred_mfccs = normalize(mfccs, axis=1)
    pred_mfccs = pred_mfccs.reshape(1, pred_mfccs.shape[0], pred_mfccs.shape[1])
    y_pred = model.predict(pred_mfccs)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    # print(y_pred_label)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: 
        print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')

# E:\nmb\nmb_data\teamvoice\clear\testvoice_F1(clear).wav 99.99927282333374 %의 확률로 남자입니다.          (x)
# E:\nmb\nmb_data\teamvoice\clear\testvoice_F1_high(clear).wav 87.52516508102417 %의 확률로 여자입니다.     (o)
# E:\nmb\nmb_data\teamvoice\clear\testvoice_F2(clear).wav 99.82030391693115 %의 확률로 여자입니다.          (o)
# E:\nmb\nmb_data\teamvoice\clear\testvoice_F3(clear).wav 97.54171371459961 %의 확률로 남자입니다.          (x)
# E:\nmb\nmb_data\teamvoice\clear\testvoice_M1(clear).wav 98.40099215507507 %의 확률로 남자입니다.          (o)
# E:\nmb\nmb_data\teamvoice\clear\testvoice_M2(claer).wav 68.71023178100586 %의 확률로 남자입니다.          (o)
# E:\nmb\nmb_data\teamvoice\clear\testvoice_M2_low(claer).wav 99.13520812988281 %의 확률로 남자입니다.      (o)

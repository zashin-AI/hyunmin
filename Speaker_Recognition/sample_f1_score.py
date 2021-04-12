# f1_score 사용하기

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
import io
import os
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K


#################### 



# 1. DATA
'''데이터 로드하기'''

#2 Modeling
'''모델 넣기'''
def modeling() :
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='elu',\
         input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = modeling()

'''지표 정의하기'''
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#3 Compile, train
'''compile 지표에 f1_m 넣기'''
model.compile(optimizer='adam', metrics=['acc', f1_m], loss='binary_crossentropy')
hist = model.fit_generator(train_generator, steps_per_epoch=len(x_train) // batch, epochs=100)

#4 Evaluate, Predict
'''evaluate 할 때 f1_m 프린트할 수 있음'''
loss, acc, f1_score = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)
print("f1_score : ", f1_score)

# loss :  0.0006527779041789472
# acc :  1.0
# f1_score :  1.0

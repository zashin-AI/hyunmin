# ml_01_data_mfcc.py
# F = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs.npy')
# print(F.shape)  # (1104, 20, 862)
# M = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs.npy')
# print(M.shape)  # (1037, 20, 862)

import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, ReLU, Softmax
from tensorflow.keras.optimizers import Adam 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs.npy')
f_lb = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs_label.npy')
m_ds = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs.npy')
m_lb = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs_label.npy')

x = np.concatenate([f_ds, m_ds], 0) # (2141, 20, 862)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (2141, 17240)
print(y.shape)  # (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1712, 17240)
print(x_test.shape)     # (429, 17240)
print(y_train.shape)    # (1712,) >  (1712, 2)
print(y_test.shape)     # (429,)  >  (429, 2)


model = Sequential()
#1차 히든레이어
model.add(Dense(256, input_shape=(1712, 17240)))
model.add(ReLU())
model.add(Dropout(0.2))
# 2차 히든 레이어
model.add(Dense(256))
model.add(ReLU())
model.add(Dropout(0.2))
# 3차 히든 레이어
model.add(Dense(256))
model.add(ReLU())
model.add(Dropout(0.2))
# 4차 히든 레이어
model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(0.001), metrics=['acc'], loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2)

# evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

# predict 데이터
pred_pathAudio = 'E:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    pred_mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=20, n_fft=512, hop_length=128)
    pred_mfcc = normalize(pred_mfcc, axis=1)
    pred_mfcc = pred_mfcc.reshape(1, pred_mfcc.shape[0] * pred_mfcc.shape[1])
    # print(pred_mels.shape)  # (1, 110336)
    y_pred = model.predict(pred_mfcc)
    # print(y_pred)
    if y_pred == 0 :                    # label 0
        print(file, '여자입니다.')
    else:                               # label 1
        print(file, '남자입니다.')


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
안된다.
loss :  0.6926737427711487
acc :  0.5221444964408875
WARNING:tensorflow:Model was constructed with shape (None, 1712, 17240) for input KerasTensor(type_spec=TensorSpec(shape=(None, 1712, 17240), dtype=tf.float32, name='dense_input'), name='dense_input', description="created by layer 'dense_input'"), but 
it was called on an input with incompatible shape (None, 17240).
Traceback (most recent call last):
  File "e:\nmb\nada\Speaker_Recognition\keras_mfcc_NN_01.py", line 107, in <module> 
    if y_pred == 0 :                    # label 0
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
'''
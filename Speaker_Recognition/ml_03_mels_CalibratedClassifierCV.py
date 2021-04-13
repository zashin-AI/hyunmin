# ml_01_data_mels.py
# F = np.load('E:/nmb/nmb_data/npy/brandnew_0_mels.npy')
# print(F.shape)  # (1104, 128, 862)
# M = np.load('E:/nmb/nmb_data/npy/brandnew_1_mels.npy')
# print(M.shape)  # (1037, 128, 862)

import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
f_ds = np.load('E:/nmb/nmb_data/npy/brandnew_0_mels.npy')
f_lb = np.load('E:/nmb/nmb_data/npy/brandnew_0_mels_label.npy')
m_ds = np.load('E:/nmb/nmb_data/npy/brandnew_1_mels.npy')
m_lb = np.load('E:/nmb/nmb_data/npy/brandnew_1_mels_label.npy')

x = np.concatenate([f_ds, m_ds], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (2141, 110336)
print(y.shape)  # (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1712, 110336)
print(x_test.shape)     # (429, 110336)
print(y_train.shape)    # (1712,)
print(y_test.shape)     # (429,)

# 모델 구성
# base_clf = GaussianNB()
# model = CalibratedClassifierCV(base_estimator=base_clf, cv=8)
# model = CalibratedClassifierCV(base_estimator=base_clf, cv=6)
# model = CalibratedClassifierCV(base_estimator=base_clf, cv=10)
# model = CalibratedClassifierCV(cv=9)
model = CalibratedClassifierCV(cv=8)    # << 시간 엄청 오래 걸리네 중단

model.fit(x_train, y_train)

# model & weight save
# pickle.dump(model, open('E:/nmb/nmb_data/cp/m03_mels_CalibratedCV.data', 'wb')) # wb : write
# pickle.dump(model, open('E:/nmb/nmb_data/cp/m03_mels_CalibratedCV6.data', 'wb')) # wb : write
# pickle.dump(model, open('E:/nmb/nmb_data/cp/m03_mels_CalibratedCV10.data', 'wb')) # wb : write
# pickle.dump(model, open('E:/nmb/nmb_data/cp/m03_mels_CalibratedCV9.data', 'wb')) # wb : write
pickle.dump(model, open('E:/nmb/nmb_data/cp/m03_mels_CalibratedCV8.data', 'wb')) # wb : write
print("== save complete ==")

# evaluate
y_pred = model.predict(x_test)
# print(y_pred[:100])
# print(y_pred[100:])

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("accuracy : \t", accuracy)
print("recall : \t", recall)
print("precision : \t", precision)

# predict 데이터
pred_pathAudio = 'E:/nmb/nmb_data/pred_voice/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    pred_mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    pred_mels = librosa.amplitude_to_db(pred_mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
    # print(pred_mels.shape)  # (1, 110336)
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    if y_pred == 0 :                    # label 0
        print(file, '여자입니다.')
    else:                               # label 1
        print(file, '남자입니다.')


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
cv=8
accuracy :       0.7249417249417249
recall :         0.6439024390243903
precision :      0.7457627118644068

E:\nmb\nmb_data\pred_voice\FY1.wav 남자입니다.                      
E:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
E:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 여자입니다.           
E:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 여자입니다.           
E:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 남자입니다.      
E:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다. 
E:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 남자입니다.      
E:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.      
E:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 여자입니다.  
정답률 7/15
time >>  0:00:18.023614
-----------------------------------------------------------------------
cv=6
accuracy :       0.7226107226107226
recall :         0.6487804878048781
precision :      0.7388888888888889

E:\nmb\nmb_data\pred_voice\FY1.wav 남자입니다.
E:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
E:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 남자입니다.
E:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 여자입니다.
정답률 5/15
time >>  0:00:14.562209
-----------------------------------------------------------------------
cv=10
accuracy :       0.7319347319347319
recall :         0.6439024390243903
precision :      0.7586206896551724

E:\nmb\nmb_data\pred_voice\FY1.wav 남자입니다.
E:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
E:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 남자입니다.   
E:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 여자입니다.       
E:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 여자입니다.   
E:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 여자입니다.
정답률 6/15
time >>  0:00:21.614832
-----------------------------------------------------------------------
cv=9
accuracy :       0.7249417249417249
recall :         0.6341463414634146
precision :      0.7514450867052023

E:\nmb\nmb_data\pred_voice\FY1.wav 남자입니다.
E:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
E:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 남자입니다.
E:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 여자입니다.
정답률 6/15
time >>  0:00:20.216947
'''
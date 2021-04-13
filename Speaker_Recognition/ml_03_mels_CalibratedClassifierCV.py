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
f_ds = np.load('C:/nmb/nmb_data/npy/brandnew_0_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/brandnew_0_mels_label.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/brandnew_1_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/brandnew_1_mels_label.npy')

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

# model = CalibratedClassifierCV(base_estimator=base_clf, cv=9)
# model = CalibratedClassifierCV(cv=8)    # << time >>  0:21:54.859041
# model = CalibratedClassifierCV(cv=6)    # << time >>  0:15:49.438282
model = CalibratedClassifierCV(cv=10)    # << time >>  0:21:54.859041

model.fit(x_train, y_train)

# model & weight save
pickle.dump(model, open('C:/nmb/nmb_data/cp/m03_mels_CalibratedCV10.data', 'wb')) # wb : write
# pickle.dump(model, open('C:/nmb/nmb_data/cp/m03_mels_CalibratedCV9.data', 'wb')) # wb : write
# pickle.dump(model, open('C:/nmb/nmb_data/cp/m03_mels_CalibratedCV8.data', 'wb')) # wb : write
# pickle.dump(model, open('C:/nmb/nmb_data/cp/m03_mels_CalibratedCV6.data', 'wb')) # wb : write
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
pred_pathAudio = 'C:/nmb/nmb_data/pred_voice/'
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
cv=8 기본값으로만 돌렸을 때,
accuracy :       0.958041958041958
recall :         0.975609756097561
precision :      0.9389671361502347
C:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.     
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.  (o)
정답률 13/15
time >>  0:21:54.859041

cv=6
accuracy :       0.958041958041958
recall :         0.975609756097561
precision :      0.9389671361502347
C:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)    
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 여자입니다. (o)
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.  
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.  (o)
정답률 14/15
time >>  0:15:49.438282

cv=10
accuracy :       0.958041958041958
recall :         0.975609756097561
precision :      0.9389671361502347
C:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.           (o)    
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.          
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.  (o)
정답률 13/15
time >>  0:28:11.623834

CV=9
accuracy :       0.958041958041958
recall :         0.975609756097561
precision :      0.9389671361502347
C:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
C:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 남자입니다.           (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 남자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 여자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 여자입니다.
C:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
C:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.  (o)
정답률 13/15
time >>  0:35:25.488474
'''
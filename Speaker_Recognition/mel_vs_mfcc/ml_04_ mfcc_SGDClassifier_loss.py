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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error
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

x = np.concatenate([f_ds, m_ds], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (2141, 17240)
print(y.shape)  # (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1712, 17240)
print(x_test.shape)     # (429, 17240)
print(y_train.shape)    # (1712,)
print(y_test.shape)     # (429,)

# 모델 구성
# model = SGDClassifier(verbose=1)
# model.fit(x_train, y_train)

# model & weight save
# pickle.dump(model, open('E:/nmb/nmb_data/cp/m04_mfccs_SGDClassifier.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
model = pickle.load(open('E:/nmb/nmb_data/cp/m04_mfccs_SGDClassifier.data', 'rb'))  # rb : read
# time >>  

# evaluate
y_pred = model.predict(x_test)
# print(y_pred[:100])
# print(y_pred[100:])

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
hamm_loss = hamming_loss(y_test, y_pred)
hinge_loss = hinge_loss(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print("accuracy : \t", accuracy)
print("recall : \t", recall)
print("precision : \t", precision)
print("f1 : \t", f1)

print("hamming_loss : \t", hamm_loss)
print("hinge_loss : \t", hinge_loss)                    # SVM에 적합한 cross-entropy
print("log_loss : \t", log_loss)                        # Cross-entropy loss와 유사한 개념
print("mse : \t", mean_squared_error(y_test, y_pred))   # Regression 모델에서의 loss


# accuracy :       0.9370629370629371
# recall :         0.926829268292683
# precision :      0.9405940594059405
# f1 :             0.9336609336609337

# hamming_loss :   0.06293706293706294
# hinge_loss :     0.585081585081585
# log_loss :       2.1737915100973217
# mse :            0.06293706293706294


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
accuracy :       0.9370629370629371
recall :         0.926829268292683
precision :      0.9405940594059405
E:\nmb\nmb_data\pred_voice\FY1.wav 여자입니다.                      (o)
E:\nmb\nmb_data\pred_voice\MZ1.wav 남자입니다.                      (o)
E:\nmb\nmb_data\pred_voice\friendvoice_F4.wav 여자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M3.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M4.wav 남자입니다.           (o)
E:\nmb\nmb_data\pred_voice\friendvoice_M5.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\friendvoice_M6.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\friendvoice_M7.wav 여자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_F1_high(clear).wav 여자입니다. (o)
E:\nmb\nmb_data\pred_voice\testvoice_F2(clear).wav 여자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_F3(clear).wav 남자입니다.
E:\nmb\nmb_data\pred_voice\testvoice_M1(clear).wav 남자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_M2(clear).wav 남자입니다.      (o)
E:\nmb\nmb_data\pred_voice\testvoice_M2_low(clear).wav 남자입니다.  (o)
정답률 10/15
time >>  0:00:03.142302
'''

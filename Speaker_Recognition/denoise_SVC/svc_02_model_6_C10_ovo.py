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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from thundersvm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error, auc
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
## data
f1 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_data0.npy')
f2 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_data1.npy')
f3 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_data2.npy')
f4 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_data3.npy')

m1 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_data0.npy')
m2 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_data1.npy')
m3 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_data2.npy')
m4 = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_data3.npy')

## label
f1_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_label0.npy')
f2_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_label1.npy')
f3_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_label2.npy')
f4_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\female_mel_label3.npy')

m1_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_label0.npy')
m2_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_label1.npy')
m3_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_label2.npy')
m4_l = np.load('E:\\nmb\\nmb_data\\npy\denoise\\male_mel_label3.npy')

x = np.concatenate([f1, f2, f3, f4, m1, m2, m3, m4], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f1_l, f2_l, f3_l, f4_l, m1_l, m2_l, m3_l, m4_l], 0)
print(x.shape)  # (3840, 110336)
print(y.shape)  # (3840,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (3072, 110336)
print(x_test.shape)     # (768, 110336)
print(y_train.shape)    # (3072,)    
print(y_test.shape)     # (768,)   

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성
model = SVC(verbose=1, C=10, random_state=42,  decision_function_shape='ovo')
model.fit(x_train, y_train)

# model & weight save
pickle.dump(model, open('E:/nmb/nmb_data/cp/svc/svc_C10_ovo.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
# model = pickle.load(open('E:/nmb/nmb_data/cp/svc/svc_C10_standard.data', 'rb'))  # rb : read
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

print("hamming_loss : \t", hamm_loss)
print("hinge_loss : \t", hinge_loss)
print("log_loss : \t", log_loss)

print("accuracy : \t", accuracy)
print("recall : \t", recall)
print("precision : \t", precision)
print("f1 : \t", f1)

# predict 데이터
pred_pathAudio = 'E:/nmb/nmb_data/predict/F'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
count_f = 0
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    pred_mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    pred_mels = librosa.amplitude_to_db(pred_mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
    pred_mels = scaler.transform(pred_mels)
    # print(pred_mels.shape)  # (1, 110336)
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    if y_pred == 0 :                    # label 0
        print(file, '여자입니다.')
        count_f += 1
    else:                               # label 1
        print(file, '남자입니다.')

pred_pathAudio = 'E:/nmb/nmb_data/predict/M'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
count_m = 0
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    pred_mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    pred_mels = librosa.amplitude_to_db(pred_mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
    pred_mels = scaler.transform(pred_mels)
    # print(pred_mels.shape)  # (1, 110336)
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    if y_pred == 0 :                    # label 0
        print(file, '여자입니다.')
    else:                               # label 1
        print(file, '남자입니다.')
        count_m += 1

print("47개 여성 목소리 중 "+str(count_f)+"개 정답입니다.")
print("48개 남성 목소리 중 "+str(count_m)+"개 정답입니다.")

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

import winsound as sd
def beepsound():
    fr = 800    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

'''
[LibSVM]..*.*
optimization finished, #iter = 3896
obj = -852.495002, rho = -1.234861
nSV = 1848, nBSV = 0
Total nSV = 1848
hamming_loss :   0.057291666666666664
hinge_loss :     0.5859375
log_loss :       1.9788017637140414
accuracy :       0.9427083333333334
recall :         0.925414364640884
precision :      0.9517045454545454
f1 :     0.938375350140056
E:\nmb\nmb_data\predict\F\F1.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F10.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F11.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F12.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F13.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F14.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F15.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F16.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F17.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F18.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F19.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F1_high.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F2.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F20.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F21.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F22.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F23.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F24.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F25.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F26.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F27.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F28.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F29.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F2_high.wav 남자입니다.
E:\nmb\nmb_data\predict\F\F2_low.wav 남자입니다.
E:\nmb\nmb_data\predict\F\F3.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F30.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F31.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F32.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F33.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F34.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F35.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F36.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F37.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F38.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F39.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F3_high.wav 남자입니다.
E:\nmb\nmb_data\predict\F\F4.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F40.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F41.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F42.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F43.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F5.wav 남자입니다.
E:\nmb\nmb_data\predict\F\F6.wav 여자입니다.
E:\nmb\nmb_data\predict\F\F7.wav 남자입니다.
E:\nmb\nmb_data\predict\F\F8.wav 남자입니다.
E:\nmb\nmb_data\predict\F\F9.wav 여자입니다.
E:\nmb\nmb_data\predict\M\M1.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M10.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M11.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M12.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M13.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M14.wav 여자입니다.
E:\nmb\nmb_data\predict\M\M15.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M16.wav 여자입니다.
E:\nmb\nmb_data\predict\M\M17.wav 여자입니다.
E:\nmb\nmb_data\predict\M\M18.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M19.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M2.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M20.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M21.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M22.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M23.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M24.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M25.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M26.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M27.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M28.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M29.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M2_high.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M2_low.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M3.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M30.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M31.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M32.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M33.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M34.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M35.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M36.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M37.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M38.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M39.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M4.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M40.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M41.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M42.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M43.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M5.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M5_high.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M5_low.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M6.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M7_high.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M7_low.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M8.wav 남자입니다.
E:\nmb\nmb_data\predict\M\M9.wav 남자입니다.
47개 여성 목소리 중 41개 정답입니다.
48개 남성 목소리 중 45개 정답입니다.
time >>  0:12:37.795257
'''
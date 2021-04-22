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
model = SVC(verbose=1, C=10, random_state=42)
# model.fit(x_train, y_train)

# model & weight save
# pickle.dump(model, open('E:/nmb/nmb_data/cp/svc/svc_C10_minmax42.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
model = pickle.load(open('E:/nmb/nmb_data/cp/svc/svc_C10_minmax42.data', 'rb'))  # rb : read
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
# pred_pathAudio = 'E:/nmb/nmb_data/predict/F'
pred_pathAudio = 'E:\\nmb\\nmb_data\\test1030\\same'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
count_f = 0
for file in files:   
    print(file)
    y, sr = librosa.load(file, sr=22050) 
    pred_mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    print(pred_mels.shape)  # (128, 5168)
    pred_mels = librosa.amplitude_to_db(pred_mels, ref=np.max)
    print(pred_mels.shape)  # (128, 5168)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
    print(pred_mels.shape)  # (1, 661504)
    # pred_mels = scaler.transform(pred_mels)

    y_pred = model.predict(pred_mels)   
    # print(y_pred)
    if y_pred == 0 :                    # label 0
        print(file, '여자입니다.')
        count_f += 1
    else:                               # label 1
        print(file, '남자입니다.')

# pred_pathAudio = 'E:/nmb/nmb_data/predict/M'
# files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
# files = np.asarray(files)
# count_m = 0
# for file in files:   
#     y, sr = librosa.load(file, sr=22050) 
#     pred_mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
#     pred_mels = librosa.amplitude_to_db(pred_mels, ref=np.max)
#     pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
#     pred_mels = scaler.transform(pred_mels)
#     # print(pred_mels.shape)  # (1, 110336)
#     y_pred = model.predict(pred_mels)
#     # print(y_pred)
#     if y_pred == 0 :                    # label 0
#         print(file, '여자입니다.')
#     else:                               # label 1
#         print(file, '남자입니다.')
#         count_m += 1

# print("47개 여성 목소리 중 "+str(count_f)+"개 정답입니다.")
# print("48개 남성 목소리 중 "+str(count_m)+"개 정답입니다.")

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

import winsound as sd
def beepsound():
    fr = 1000    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

'''

'''
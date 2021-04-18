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
# from thundersvm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error, auc
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기 (denoise 안 한 데이터)
## data
f1 = np.load('E:\\nmb\\nmb_data\\npy\\balance\\balance_f_mels.npy')
m1 = np.load('E:\\nmb\\nmb_data\\npy\\balance\\balance_m_mels.npy')

## label
f1_l = np.load('E:\\nmb\\nmb_data\\npy\\balance\\balance_f_label_mels.npy')
m1_l = np.load('E:\\nmb\\nmb_data\\npy\\balance\\balance_m_label_mels.npy')


x = np.concatenate([f1, m1], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f1_l, m1_l], 0)
print(x.shape)  # (3840, 110336)
print(y.shape)  # (3840,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (3072, 110336)
print(x_test.shape)     # (768, 110336)
print(y_train.shape)    # (3072,)    
print(y_test.shape)     # (768,)   

# 모델 구성
model = SVC(verbose=1)
model.fit(x_train, y_train)

# model & weight save
pickle.dump(model, open('E:/nmb/nmb_data/cp/svc/svc_base_raw.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
# model = pickle.load(open('E:/nmb/nmb_data/cp/svc/svc_base_raw.data', 'rb'))  # rb : read
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

'''
[LibSVM].*.*
optimization finished, #iter = 2383
obj = -697.182752, rho = -1.243500
nSV = 1711, nBSV = 647
Total nSV = 1711
hamming_loss :   0.08723958333333333
hinge_loss :     0.6158854166666666
log_loss :       3.013174490097358
accuracy :       0.9127604166666666
recall :         0.8839779005524862
precision :      0.927536231884058
f1 :     0.9052333804809052
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
E:\nmb\nmb_data\predict\F\F2_high.wav 여자입니다.
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
E:\nmb\nmb_data\predict\F\F7.wav 여자입니다.
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
E:\nmb\nmb_data\predict\M\M31.wav 여자입니다.
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
47개 여성 목소리 중 43개 정답입니다.
48개 남성 목소리 중 44개 정답입니다.
time >>  0:09:21.036778
'''
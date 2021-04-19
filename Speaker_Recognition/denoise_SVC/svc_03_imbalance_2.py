import numpy as np
import datetime 
import librosa
import sklearn
import os
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
# [1] 여 < 남
# f1 = np.load('C:\\nmb\\nmb_data\\npy\\imbalance\\imbalance_dataset1_denoise_f_mels.npy')
# m1 = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_mels.npy')
# print(f1.shape, m1.shape)
# ## label
# f1_l = np.load('C:\\nmb\\nmb_data\\npy\\imbalance\\imbalance_dataset1_denoise_f_label_mels.npy')
# m1_l = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_label_mels.npy')

## data
# [2] 여 > 남
f1 = np.load('C:\\nmb\\nmb_data\\npy\\balance\\balance_f_mels.npy')
m1 = np.load('C:\\nmb\\nmb_data\\npy\\imbalance\\imbalance_dataset2_denoise_m_mels.npy')
print(f1.shape, m1.shape)
## label
f1_l = np.load('C:\\nmb\\nmb_data\\npy\\balance\\balance_f_label_mels.npy')
m1_l = np.load('C:\\nmb\\nmb_data\\npy\imbalance\\imbalance_dataset2_denoise_m_label_mels.npy')

x = np.concatenate([f1, m1], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f1_l, m1_l], 0)
print(x.shape)  # (2400, 110336)
print(y.shape)  # (2400,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1920, 110336)
print(x_test.shape)     # (480, 110336)
print(y_train.shape)    # (1920,)   
print(y_test.shape)     # (480,)  

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성
model = SVC(verbose=1, C=10, random_state=42)
model.fit(x_train, y_train)

# model & weight save
# pickle.dump(model, open('C:\\nmb\\nmb_data\\cp\svc/svc_imbalance1.data', 'wb')) # wb : write
pickle.dump(model, open('C:\\nmb\\nmb_data\\cp\svc/svc_imbalance2.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
# model = pickle.load(open('E:/nmb/nmb_data/cp/svc/svc_imbalance1.data', 'rb'))  # rb : read
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
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
        pred_mels = scaler.transform(pred_mels)
        y_pred = model.predict(pred_mels)
        # y_pred_label = np.argmax(y_pred)
        if y_pred == 0 :  # 여성이라고 예측
            print(file, '여자입니다.')
            if length > 9 :    # 이상치
                if name == 'F' :
                    count_odd = count_odd + 1                   
            else :
                if name == 'F' :
                    count_f = count_f + 1
                
        else:                   # 남성이라고 예측              
            print(file, '남자입니다.')
            if length > 9 :    # 이상치
                if name == 'M' :
                    count_odd = count_odd + 1
            else :
                if name == 'M' :
                    count_m = count_m + 1
                
                    
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")
print("10개 이상치 목소리 중 "+str(count_odd)+"개 정답")

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >


'''
# [1] 여 < 남
optimization finished, #iter = 2175
obj = -490.381229, rho = -0.434290
nSV = 1085, nBSV = 1
Total nSV = 1085
[LibSVM]hamming_loss :   0.05416666666666667
hinge_loss :     0.27291666666666664
log_loss :       1.8708853704452337
accuracy :       0.9458333333333333
recall :         0.9866666666666667
precision :      0.9462915601023018
f1 :     0.9660574412532638
C:\nmb\nmb_data\predict\F\F1.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F10.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F11.wav 남자입니다.
C:\nmb\nmb_data\predict\F\F12.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F13.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F14.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F15.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F16.wav 남자입니다.
C:\nmb\nmb_data\predict\F\F17.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F18.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F19.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F2.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F20.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F21.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F22.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F23.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F24.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F25.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F26.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F27.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F28.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F29.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F3.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F30.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F31.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F32.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F33.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F34.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F35.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F36.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F37.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F38.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F39.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F4.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F40.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F41.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F42.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F43.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F5.wav 남자입니다.
C:\nmb\nmb_data\predict\F\F6.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F7.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F8.wav 남자입니다.
C:\nmb\nmb_data\predict\F\F9.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M1.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M10.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M11.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M12.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M13.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M14.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M15.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M16.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M17.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M18.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M19.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M2.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M20.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M21.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M22.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M23.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M24.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M25.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M26.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M27.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M28.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M29.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M3.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M30.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M31.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M32.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M33.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M34.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M35.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M36.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M37.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M38.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M39.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M4.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M40.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M41.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M42.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M43.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M5.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M6.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M8.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M9.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\F1_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\F2_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\F2_low.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\F3_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_low.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\M5_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_low.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\M7_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_low.wav 여자입니다.
43개 여성 목소리 중 38개 정답
42개 남성 목소리 중 29개 정답
10개 이상치 목소리 중 3개 정답
time >>  0:03:13.003068

# [2] 여 > 남
optimization finished, #iter = 2249
obj = -407.399082, rho = -1.022774
nSV = 1158, nBSV = 0
Total nSV = 1158
[LibSVM]hamming_loss :   0.07291666666666667
hinge_loss :     0.8791666666666667
log_loss :       2.5184641062580946
accuracy :       0.9270833333333334
recall :         0.6989247311827957
precision :      0.9027777777777778
f1 :     0.7878787878787878
C:\nmb\nmb_data\predict\F\F1.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F10.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F11.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F12.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F13.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F14.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F15.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F16.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F17.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F18.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F19.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F2.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F20.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F21.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F22.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F23.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F24.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F25.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F26.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F27.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F28.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F29.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F3.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F30.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F31.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F32.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F33.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F34.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F35.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F36.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F37.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F38.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F39.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F4.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F40.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F41.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F42.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F43.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F5.wav 남자입니다.
C:\nmb\nmb_data\predict\F\F6.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F7.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F8.wav 여자입니다.
C:\nmb\nmb_data\predict\F\F9.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M1.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M10.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M11.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M12.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M13.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M14.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M15.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M16.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M17.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M18.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M19.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M2.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M20.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M21.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M22.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M23.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M24.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M25.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M26.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M27.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M28.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M29.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M3.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M30.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M31.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M32.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M33.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M34.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M35.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M36.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M37.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M38.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M39.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M4.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M40.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M41.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M42.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M43.wav 여자입니다.
C:\nmb\nmb_data\predict\M\M5.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M6.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M8.wav 남자입니다.
C:\nmb\nmb_data\predict\M\M9.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\F1_high.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\F2_high.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\F2_low.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\F3_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M2_low.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M5_high.wav 여자입니다.
C:\nmb\nmb_data\predict\ODD\M5_low.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_high.wav 남자입니다.
C:\nmb\nmb_data\predict\ODD\M7_low.wav 남자입니다.
43개 여성 목소리 중 42개 정답
42개 남성 목소리 중 32개 정답
10개 이상치 목소리 중 8개 정답
time >>  0:03:19.210109
'''
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error
# from sklearn.utils import all_estimators  
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
x = np.load('E:\\nmb\\nmb_data\\5s_last_0510\\total_data.npy')
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

y = np.load('E:\\nmb\\nmb_data\\5s_last_0510\\total_label.npy')
print(x.shape)  # (4536, 110336)
print(y.shape)  # (4536,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (3628, 110336)
print(x_test.shape)     # (908, 110336)
print(y_train.shape)    # (3628,)
print(y_test.shape)     # (908,)

# 모델 구성
model = SVC(verbose=1)
hist = model.fit(x_train, y_train)

# model & weight save
pickle.dump(model, open('E:\\nmb\\nmb_data\\cp\\5s_last_0510_ml\\SVC_1_default.data', 'wb')) # wb : write
print("== save complete ==")

# model load
# model = pickle.load(open('E:\\nmb\\nmb_data\\cp\\5s_last_0510_ml\\SVC_1_default.data', 'rb'))  # rb : read
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

pred = ['E:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\F', 'E:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
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
        # print(pred_mels.shape)
        y_pred = model.predict(pred_mels)
        # print(y_pred)
        if y_pred == 0:   # 여성이라고 예측
            print(file, '여자입니다.')
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '남자입니다.')
            if name == 'M' :
                count_m = count_m + 1

print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
optimization finished, #iter = 2550
obj = -671.040281, rho = -1.081500
nSV = 1780, nBSV = 602
Total nSV = 1780
[LibSVM]== save complete ==
accuracy :       0.9427312775330396
recall :         0.928921568627451
precision :      0.9427860696517413
f1 :     0.9358024691358023
hamming_loss :   0.05726872246696035
hinge_loss :     0.6079295154185022
log_loss :       1.9780118538284697
mse :    0.05726872246696035
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F1.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F10.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F11.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F12.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F13.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F14.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F15.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F16.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F17.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F18.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F19.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F2.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F20.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F21.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F22.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F23.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F24.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F25.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F26.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F27.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F28.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F29.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F3.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F30.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F31.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F32.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F33.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F34.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F35.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F36.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F37.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F38.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F39.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F4.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F40.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F41.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F42.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F43.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F5.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F6.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F7.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F8.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F9.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M1.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M10.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M11.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M12.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M13.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M14.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M15.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M16.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M17.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M18.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M19.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M2.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M20.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M21.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M22.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M23.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M24.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M25.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M26.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M27.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M28.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M29.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M3.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M30.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M31.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M32.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M33.wav 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M34.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M35.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M36.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M37.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M38.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M39.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M4.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M40.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M41.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M42.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M43.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M5.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M6.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M7.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M8.wav 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M9.wav 남자입니다.
43개 여성 목소리 중 39개 정답
43개 남성 목소리 중 38개 정답
time >>  0:11:53.657942

'''
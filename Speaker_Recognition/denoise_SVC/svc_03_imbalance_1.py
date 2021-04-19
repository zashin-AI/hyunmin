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
f1 = np.load('C:\\nmb\\nmb_data\\npy\\imbalance\\imbalance_dataset1_denoise_f_mels.npy')
m1 = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_mels.npy')
print(f1.shape, m1.shape)
## label
f1_l = np.load('C:\\nmb\\nmb_data\\npy\\imbalance\\imbalance_dataset1_denoise_f_label_mels.npy')
m1_l = np.load('C:\\nmb\\nmb_data\\npy\\denoise\\denoise_balance_m_label_mels.npy')

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
pickle.dump(model, open('C:\\nmb\\nmb_data\\cp\svc/svc_imbalance1.data', 'wb')) # wb : write
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

'''
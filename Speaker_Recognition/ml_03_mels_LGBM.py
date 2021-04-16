# ml_01_data_mels.py
# F = np.load('E:/nmb/nmb_data/npy/brandnew_0_mels.npy')
# print(F.shape)  # (1104, 128, 862)
# M = np.load('E:/nmb/nmb_data/npy/brandnew_1_mels.npy')
# print(M.shape)  # (1037, 128, 862)

import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error, auc
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
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
model = LGBMClassifier(n_estimators=10000, device='gpu')
model = XGBClassifier(n_jobs = -1, use_label_encoder=False,
                    n_estimators=10000, 
                    tree_method = 'gpu_hist',
                    # predictor='gpu_predictor'
                    predictor='cpu_predictor',
                    gpu_id=0
)
model.fit(x_train, y_train)

# model & weight save
pickle.dump(model, open('E:/nmb/nmb_data/cp/m03_mels_LGBMClassifier.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
# model = pickle.load(open('E:/nmb/nmb_data/cp/m03_mels_LGBMClassifier.data', 'rb'))  # rb : read
# time >>  0:30:49.704071

# evaluate
y_pred = model.predict(x_test)
# print(y_pred[:100])
# print(y_pred[100:])

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("accuracy : \t", accuracy)
print("recall : \t", recall)
print("precision : \t", precision)
print("f1 : \t", f1)

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
model = LGBMClassifier(n_estimators=10000)

'''
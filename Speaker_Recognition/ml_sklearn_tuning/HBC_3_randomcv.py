import numpy as np
import datetime 
import librosa
import sklearn
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.svm import NuSVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier , HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error
# from sklearn.utils import all_estimators 
import pickle  
import warnings
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
x = np.load('C:\\nmb\\nmb_data\\5s_last_0510\\total_data.npy')
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

y = np.load('C:\\nmb\\nmb_data\\5s_last_0510\\total_label.npy')
print(x.shape)  # (4536, 110336)
print(y.shape)  # (4536,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (3628, 110336)
print(x_test.shape)     # (908, 110336)
print(y_train.shape)    # (3628,)
print(y_test.shape)     # (908,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성
model = HistGradientBoostingClassifier(verbose=1, random_state=42)
parameters = {"learning_rate": sp_randFloat(),
              "max_iter"    : [1000,1200,1500],
              "l2_regularization" : [1.5, 0.5, 0, 1],
              "max_depth"    : sp_randInt(4, 10)
            }
randm = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                            cv = 2, n_iter = 10, n_jobs=-1)
randm.fit(x_train, y_train)

print(" Results from Random Search " )
print("The best estimator across ALL searched params:", randm.best_estimator_)
print("The best score across ALL searched params:", randm.best_score_)
print(" The best parameters across ALL searched params:", randm.best_params_)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

import winsound as sd
def beepsound():
    fr = 1000    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()
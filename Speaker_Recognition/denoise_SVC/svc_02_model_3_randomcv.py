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
from sklearn.preprocessing import StandardScaler
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

# 모델 구성
svc = SVC(verbose=1, random_state=42)

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_grid = [{'C': param_range, 'gamma': param_range, 'kernel': ['linear']}, 
            {'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}]

model = RandomizedSearchCV(svc, param_grid, scoring='accuracy', n_jobs=-1)

model.fit(x_train, y_train)

print(model.best_score_)

print(model.best_params_)

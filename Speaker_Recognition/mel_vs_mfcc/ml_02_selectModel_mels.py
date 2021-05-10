# ml_01_data_mels.py
# F = np.load('E:/nmb/nmb_data/npy/brandnew_0_mels.npy')
# print(F.shape)  # (1104, 128, 862)
# M = np.load('E:/nmb/nmb_data/npy/brandnew_1_mels.npy')
# print(M.shape)  # (1037, 128, 862)

import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import all_estimators    
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

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :    # 분류형 모델 전체를 반복해서 돌린다.
    # try ... except... : 예외처리 구문
    try :   # 에러가 없으면 아래 진행됨
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 



allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :    
    try :   
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :          
        print(name, "은 없는 모델") 

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
* : 0.95 이상   >>  CalibratedClassifierCV, LinearDiscriminantAnalysis, LinearSVC, LogisticRegressionCV, SVC
+ : 0.94 이상   >>  GradientBoostingClassifier, HistGradientBoostingClassifier, LogisticRegression, NuSVC, RidgeClassifier, RidgeClassifierCV
! : 0.93 이상   >>  PassiveAggressiveClassifier, Perceptron

AdaBoostClassifier 의 정답률 :              0.9137529137529138
BaggingClassifier 의 정답률 :               0.8927738927738927
BernoulliNB 의 정답률 :                     0.5221445221445221
CalibratedClassifierCV 의 정답률 :          0.958041958041958   *
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 은 없는 모델
DecisionTreeClassifier 의 정답률 :          0.8461538461538461
DummyClassifier 의 정답률 :                 0.5221445221445221
ExtraTreeClassifier 의 정답률 :             0.8484848484848485
ExtraTreesClassifier 의 정답률 :            0.9277389277389277
GaussianNB 의 정답률 :                      0.7319347319347319
GaussianProcessClassifier 의 정답률 :       0.5221445221445221
GradientBoostingClassifier 의 정답률 :      0.9417249417249417  +
HistGradientBoostingClassifier 의 정답률 :  0.9440559440559441  +
KNeighborsClassifier 의 정답률 :            0.916083916083916
LabelPropagation 의 정답률 :                0.5221445221445221
LabelSpreading 의 정답률 :                  0.5221445221445221
LinearDiscriminantAnalysis 의 정답률 :      0.951048951048951   *
LinearSVC 의 정답률 :                       0.9557109557109557  *
LogisticRegression 의 정답률 :              0.9463869463869464  +
LogisticRegressionCV 의 정답률 :            0.958041958041958   *
MLPClassifier 의 정답률 :                   0.5221445221445221
MultiOutputClassifier 은 없는 모델
MultinomialNB 은 없는 모델
NearestCentroid 의 정답률 :                 0.8881118881118881
NuSVC 의 정답률 :                           0.9417249417249417  +
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :     0.9370629370629371  !
Perceptron 의 정답률 :                      0.9324009324009324  !
QuadraticDiscriminantAnalysis 의 정답률 :   0.5594405594405595
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :          0.9230769230769231
RidgeClassifier 의 정답률 :                 0.9487179487179487  +
RidgeClassifierCV 의 정답률 :               0.9487179487179487  +
SGDClassifier 의 정답률 :                   0.916083916083916
SVC 의 정답률 :                             0.951048951048951   *
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델

time >>  1:41:08.493187
'''
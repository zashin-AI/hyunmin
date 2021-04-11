# ml_01_data_mfcc.py
# F = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs.npy')
# print(F.shape)  # (1104, 20, 862)
# M = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs.npy')
# print(M.shape)  # (1037, 20, 862)

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
f_ds = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs.npy')
f_lb = np.load('E:/nmb/nmb_data/npy/brandnew_0_mfccs_label.npy')
m_ds = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs.npy')
m_lb = np.load('E:/nmb/nmb_data/npy/brandnew_1_mfccs_label.npy')


x = np.concatenate([f_ds, m_ds], 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)  # (2141, 17240)
print(y.shape)  # (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (1712, 17240)
print(x_test.shape)     # (429, 17240)
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

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
* : 0.94 이상 >> LinearDiscriminantAnalysis, LogisticRegressionCV, SVC
+ : 0.93 이상 >> ExtraTreesClassifier, GradientBoostingClassifier, NuSVC, SGDClassifier
! : 0.92 이상 >> CalibratedClassifierCV, ComplementNB, GaussianNB, LogisticRegression, MultinomialNB, NearestCentroid, PassiveAggressiveClassifier, Perceptron

AdaBoostClassifier 의 정답률 :              0.916083916083916
BaggingClassifier 의 정답률 :               0.8717948717948718
BernoulliNB 의 정답률 :                     0.5897435897435898
CalibratedClassifierCV 의 정답률 :          0.9254079254079254  !
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :                    0.9230769230769231  !
DecisionTreeClassifier 의 정답률 :          0.7855477855477856
DummyClassifier 의 정답률 :                 0.5221445221445221
ExtraTreeClassifier 의 정답률 :             0.696969696969697
ExtraTreesClassifier 의 정답률 :            0.9324009324009324  +
GaussianNB 의 정답률 :                      0.9207459207459208  !
GaussianProcessClassifier 의 정답률 :       0.8344988344988346
GradientBoostingClassifier 의 정답률 :      0.9347319347319347  +
HistGradientBoostingClassifier 의 정답률 :  0.9184149184149184
KNeighborsClassifier 의 정답률 :            0.9090909090909091
LabelPropagation 의 정답률 :                0.5221445221445221
LabelSpreading 의 정답률 :                  0.5221445221445221
LinearDiscriminantAnalysis 의 정답률 :      0.9417249417249417  *
LinearSVC 의 정답률 :                       0.916083916083916
LogisticRegression 의 정답률 :              0.9277389277389277  !
LogisticRegressionCV 의 정답률 :            0.9417249417249417  *
MLPClassifier 의 정답률 :                   0.5221445221445221
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :                   0.9207459207459208  !
NearestCentroid 의 정답률 :                 0.9230769230769231  !
NuSVC 의 정답률 :                           0.9393939393939394  +
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :     0.9254079254079254  !
Perceptron 의 정답률 :                      0.9230769230769231  !
QuadraticDiscriminantAnalysis 의 정답률 :   0.5454545454545454
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :          0.916083916083916
RidgeClassifier 의 정답률 :                 0.8764568764568764
RidgeClassifierCV 의 정답률 :               0.8811188811188811
SGDClassifier 의 정답률 :                   0.9370629370629371  +
SVC 의 정답률 :                             0.9463869463869464  *
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''
Melspectrogram 데이터를 allAlgorithms = all_estimators(type_filter='classifier') 로 돌려봤을 때 가장 좋게 나왔던 모델이다. 
```python
0.95 이상 >  CalibratedClassifierCV, LinearDiscriminantAnalysis, LinearSVC, LogisticRegressionCV, SVC    
0.94 이상 >  GradientBoostingClassifier, HistGradientBoostingClassifier, LogisticRegression, NuSVC, RidgeClassifier, RidgeClassifierCV    
0.93 이상 >  PassiveAggressiveClassifier, Perceptron    
```

이 중에서 정답률이 높았던 모델을 더 자세히 조사해보았다.    

# 1. Sklearn.svm.SVC
```python
model = sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
                    shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, 
                    verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
```
* C-Support Vector Classification
* libsvm으로부터 나온 모델

  + **SVM (Support Vector Machine)**
  + 데이터 분석 중 분류에 이용되며 지도 학습 방식의 모델
  + **결정 경계** : 분류를 위한 기준 선을 정의하는 모델 , 분류되지 않은 새로운 점이 나타나면 경계의 어느 쪽에 속하는지 확인해서 분류 과제를 수행할 수 있다.
  + **Support Vectors** : 결정 경계와 가까이 있는 데이터 포인트들을 의미함
  + 최적의 결정 경계 : 마진(점선으로부터 결정 경계까지의 거리)을 최대화하는 것 , 두 점들 사이의 경계가 뚜렷하다는 것을 의미함 
* 멀티클래스 지원은 일대일 체계에 따라 처리한다( The multiclass support is handled according to a one-vs-one scheme. )
* C : 오류를 어느 정도 허용할 것인지
  + C 값이 클수록 : 오류 허용 안함, 포인터와 결정 경계 사이의 거리가 매우 좁다. 오버피팅의 문제가 생길 수 있다.
  + C 값이 작을수록 : 오류 허용함, 포인터와 결정 경계 사이의 거리가 넓다. 언더피팅 문제가 발생할 수 있다.
* kernel : 직선을 어떤 모양으로 그릴 것인지
  + linear : 선형
  + poly : 다항식, 데이터를 더 높은 차원으로 변형하여 나타냄
  + rbf : Radial Bias Function, 2차원의 점을 무한한 차원의 점으로 변환
* gamma : 결정 경계를 얼마나 유연하게 그을 것인지
  + gamma 높으면 : 결정 경계를 구불구불 긋는다. 오버피팅을 초래할 수 있음
  + gamma 낮추면 : 결정 경계를 직선에 가깝게 긋는다. 언더피팅을 초래할 수 있음
# 2. NuSVC
```python
model = sklearn.svm.NuSVC(*, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
                          shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
                          verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
```

# 3. GradientBosstingClassifier

# 4. HistGradientBosstingClassifier

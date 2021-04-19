Melspectrogram 데이터를 _allAlgorithms = all_estimators(type_filter='classifier')_ 로 돌려봤을 때 가장 좋게 나왔던 모델이다. 
```python
0.95 이상 >  CalibratedClassifierCV, LinearDiscriminantAnalysis, LinearSVC, LogisticRegressionCV, SVC    
0.94 이상 >  GradientBoostingClassifier, HistGradientBoostingClassifier, LogisticRegression, NuSVC, RidgeClassifier, RidgeClassifierCV    
0.93 이상 >  PassiveAggressiveClassifier, Perceptron    
```

이 중에서 정답률이 높았던 모델을 더 자세히 조사해보았다.    

# 1. SVC
[공식문서](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
```python
model = sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
                    shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, 
                    verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
```
* C-Support Vector Classification
* libsvm으로부터 나온 모델
![이미지 001](https://user-images.githubusercontent.com/70581043/114386231-ea823400-9bcb-11eb-852f-c9ea4085df74.png)
  + **SVM (Support Vector Machine)**
  + 데이터 분석 중 분류에 이용되며 지도 학습 방식의 모델
  + **결정 경계** : 분류를 위한 기준 선을 정의하는 모델 , 분류되지 않은 새로운 점이 나타나면 경계의 어느 쪽에 속하는지 확인해서 분류 과제를 수행할 수 있다.
  + **Support Vectors** : 결정 경계와 가까이 있는 데이터 포인트들을 의미함
  + 최적의 결정 경계 : 마진(점선으로부터 결정 경계까지의 거리)을 최대화하는 것 , 두 점들 사이의 경계가 뚜렷하다는 것을 의미함 
* 멀티클래스 지원은 일대일 체계에 따라 처리한다( The multiclass support is handled according to a one-vs-one scheme. )
* C : 오류를 어느 정도 허용할 것인지
  ![이미지 002](https://user-images.githubusercontent.com/70581043/114386290-fcfc6d80-9bcb-11eb-90d8-784ebb136b13.png)

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
[공식문서](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)
```python
model = sklearn.svm.NuSVC(*, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
                          shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
                          verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
```
* Nu-Support Vector Classification
* SVC와 비슷하지만  null support vectors의 개수를 조절한다.
* 파라미터 nu (default=0.5) : 서포트 벡터를 조절한다.
  + 0과 1사이로 지정된다.
  + bound on the fraction of margin errors and a lower bound of the fraction of support vectors (마진 에러 & 서포트 벡터 부분) (링크)
  + 예) 0.05라면 5% 가 잘못 분류되는 것이면서 & 5%가 최소 서포트 벡터임.

# 3. GradientBoostingClassifier
[공식문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
```python
model = sklearn.ensemble.GradientBoostingClassifier(*, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                                    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                                    min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                                    min_impurity_split=None, init=None, random_state=None, max_features=None,
                                                    verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
                                                    n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
```
* Gradient Boosting for classification
* 여러 개의 decision tree를 묶어 강력한 model을 만드는 ensemble기법
* 이전 tree의 오차를 보완하는 방식으로 tree를 만듭니다.
* 랜덤 포레스트처럼 tree를 여러 개 만든다. 
   + 단, 한꺼번에 tree를 만들지 않고 tree를 하나 만든 다음 그것의 오차를 줄이는 방법으로 tree를 만드는 단계로 진행한다.
* gradient boosting : 무작위성이 없어 powerful한 pre-pruning이 사용되며 1~5 정도 깊이의 tree를 사용하므로 메모리를 적게 사용하고 예측도 빠릅니다. gradient boosting은 이런 얕은 트리들을 계속해서 연결해나가는 것 [링크](https://woolulu.tistory.com/30)
* criterion=’Friedman mse’ : mse 업그레이드 버전, mse 나 mae 보다 더 나은 근사치를 제공할 수 있기 때문에 일반적으로 더 좋다. [링크](https://wikidocs.net/26289)
* 파라미터 n_estimators  : tree의 개수
  + 주로 깊이를 작게 하고 tree의 개수를 늘리는 전략을 많이 취한다.
* 파라미터 learning rate : 값이 클수록 복잡한 모델을 만든다. 이전에 만든 tree의 오류에 기반하여 얼마나 많이 수정해 나갈지의 비율을 의미한다. [링크](https://jfun.tistory.com/122)
* boosting : 알고리즘을 학습하면서 tree를 더해가는 과정  

# 4. HistGradientBoostingClassifier
[공식문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
```python
model = sklearn.ensemble.HistGradientBoostingClassifier(loss='auto', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
                                                        max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255,
                                                        categorical_features=None, monotonic_cst=None, warm_start=False,
                                                        early_stopping='auto', scoring='loss', validation_fraction=0.1,
                                                        n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
```
* Histogram-based Gradient Boosting Classification Tree
* Gradient Boosting Classification Tree보다 큰 데이터들을 빠르게 분석할 수 있다.
* 훈련을 하는 동안 결측치가 있을 때, 왼쪽 또는 오른쪽 child로 이동해야 하는지 여부를 각 분학 지점에서 학습을 한다.
*  gradient boosting ensemble 에서 보다 빠르게 결정 트리를 훈련시킬 수 있다. (training faster decision trees used in the gradient boosting ensemble.)

# 5. LogisticRegressionCV
[공식문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)

```python
model = sklearn.linear_model.LogisticRegressionCV(*, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', 
                                                  scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, 
                                                  n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None)
```
* Logistic Regression CV classifier.
* 이진분류 모델로 많이 사용된다.
* 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0과 1사이의 값으로 예측하고 그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류해주는 지도 학습 알고리즘 [링크](http://hleecaster.com/ml-logistic-regression-concept/)    
    
# 6. XGB, LightXGB, CatBoost
![1_E006sjlIjabDJ3jNixRSnA](https://user-images.githubusercontent.com/70581043/115263875-c1841500-a170-11eb-8bda-3f4b4a7389e1.png)

### XGB
* 샘플링 기술을 사용하지 않고, 순차적으로 모든 데이터들을 학습시킨다.
* 학습이 느리다.
### LightXGB
* one-side sampling
* 의사 결정 트리의 정확성을 유지하며 속도를 높인다.
* 큰 데이터를 분리해가며 하위 단계로 내려간다.
* 단점 : 과적합 발생
* 주의 : 트리 깊이를 제어하는 것이 중요하다.
### CatBoost
* Category Boosting
* Minimal Variance Sampling
* 샘플링이 분학 수준이 아닌 트리 수준에서 발생, 트리가 옆으로 넓어진다.
* 모든 레벨의 노드에 동등하게 적용된다.
* 기존 부스팅 모델이 모든 훈련 데이터를 대상으로 잔차를 계산했다면, catbosst는 일부만 잔차 계산을 한 뒤 모델에 적용
* 단점 : sparse한 데이터는 처리하지 못한다. 데이터가 수치형 변수인 경우 LGBM보다 학습이 느리다.
[참고자료](https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc)


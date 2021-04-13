머신러닝이 잘 작동하는지 통계적으로 확인해본다.    
![99DC064C5BE056CE10](https://user-images.githubusercontent.com/70581043/114490303-8f922080-9c4f-11eb-87c5-ed15bf4d7189.png)

예시 상황 : 눈이 오는 날을 맞추는 모델을 만들어다. 이 모델의 성능을 알아보자    

# 1. Accuracy
* 예시 ) 실제 눈이 내린 날을 맞춘 값
* TP / TP + FP + FN + TN
* 전체 데이터 중에서 올바르게 예측된 데이터의 수를 나눈다.
* 문제점 : 데이터들이 불균형하게 있는 경우, 제대로 측정하지 못한다.
  + 365일 중에서 눈이 내린 날이 적기 때문에 항상 False로 예측해도 예측을 잘한 모델이 된다.
  
# 2. Recall
* 예시 ) 모델이 눈이 내릴거라 예측한 날의 수를 실제로 눈이 내린 날의 수로 나눈 값
* TP / TP + FN
* 실제 True 인 데이터를 모델이 True라고 인식한 데이터의 수
![recall](https://user-images.githubusercontent.com/70581043/114492009-a5edab80-9c52-11eb-9e04-cf241bc27c8a.jpg)

# 3. Precision
* 예시 ) 실제로 눈이 내린 날의 수를 모델이 눈이 내릴거라 예측한 날의 수로 나눈 값
* TP / TP + FP
* 모델이 True라고 예측한 데이터 중 실제로 True 인 데이터의 수
![precision](https://user-images.githubusercontent.com/70581043/114492255-2dd3b580-9c53-11eb-8740-90c970d2e146.jpg)

# 4. F1 score
* recall score와 precision score 두 지표를 하나의 지표로 만들어준다.
* recall, precision의 조화평균   
![이미지 001](https://user-images.githubusercontent.com/70581043/114491407-8efa8980-9c51-11eb-8179-9ea8321b24ce.png)

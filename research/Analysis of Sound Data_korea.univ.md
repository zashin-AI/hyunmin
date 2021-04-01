Analysis of Sound Data
=====================
출처 : 김성범[ 단장 / 4단계 BK21 산업경영공학교육연구단 ] 유튜브 영상 [(🔗링크)](https://youtu.be/Z_6tAxb89sw)

소리 데이터의 특징과 음성 데이터를 어쪼고 저쩌고 요약하기


# 소리 데이터 개념 및 원리
### Sound Data
* 소리 : 공기나 물 같은 매질의 진동을 통해 전달되는 종파, 사람의 귀에 들려오는 소리는 공기 속을 전해오는 파동
* 소리의 3요소 : 세기, 높낮이, 음색
* 소리의 샘플링 레이트 (Sampling Rate)과정 
  + 소리를 컴퓨터에 입력시키기 위해 음파를 숫자로 표현할 필요가 있음
  + 샘플링 레이트 (Sampling Rate) : 아날로그 소리를 디지털로 변환 시킨 것
  + 표준 샘플링 레이트 (Sampling Rate) 44.1kHz → 소리로부터 1초당 44100 개의 샘플을 추출했다는 것을 의미함
   ![이미지 001](https://user-images.githubusercontent.com/70581043/113291431-fbaa8580-932d-11eb-9267-f4aef93f99bb.png)

# 데이터 전처리 및 특징 추출 기법
### Feature Engineering
* 파동 : 시간과 주파수로 구성되어 있음   
![이미지 003](https://user-images.githubusercontent.com/70581043/113291560-2bf22400-932e-11eb-9038-122d6443f638.png)
* 파장 : 무한 개의 코사인 함수로 이루어져 있다. 주기함수의 합으로 나타나져 있다. 다른 주기의 주파수 성분의 합으로 이루어져 있다.   
![이미지 004](https://user-images.githubusercontent.com/70581043/113291617-3dd3c700-932e-11eb-890f-f04dae8ccbb5.png)
* 소리 특성 정보를 추출하기 위한 다양한 특징 추출 방법론이 존재함 (Spectrum, Mel Spectrum, MFCC 등)

### 1. 스펙트럼(Spectrum)

* 파동의 시간 영역을 주파수 영역으로 변환 (단점:시간 정보가 없다)
* 음향 신호를 주파수, 진폭으로 분석하여 보여줌
* 고속 푸리에 변환을 적용
* x축 : 주파수, y축 : 진폭    
![이미지 005](https://user-images.githubusercontent.com/70581043/113291654-4a581f80-932e-11eb-880e-e6507708d987.png)


### 2. 멜 스펙트럼(Mel Spectrum)

* 주파수 특성이 시간에 따라 달라지는 오디오를 분석하기 위한 특징 추출 기법
* 스펙트럼에 mel scale을 적용한 것    
![이미지 006](https://user-images.githubusercontent.com/70581043/113291689-55ab4b00-932e-11eb-84ca-e831fea3ba09.png)
* 고주파로 갈 수록 사람이 구분하는 주파수 간격이 넓어지는 데 이를 반영해줌
* 시간 정보가 어떻게 반영되는가 ? 
  + 시간에 따라 프레임 별로 나눈다. > 프레임 안에 있는 주파수를 분석해서 스펙트럼을 생성한다. > 조금씩 프레임을 이동한다. > 스펙트럼을 합친다. > mel scale을 적용한다.
  + 어느 정도 길이만큼 프레임 길이를 자를 것인지, 슬라이딩 범위를 하이퍼 파라미터로 설정해야 한다.    
![이미지 008](https://user-images.githubusercontent.com/70581043/113291767-75427380-932e-11eb-8e63-4b9d6dbdaf8f.png)

### 3. MFCC (Mel-Frequency Cepstral Coefficient)
* 멜 스펙트럼에서 켑스트럴(Cepstral) 분석을 통해 추출된 값
  + Cepstral : 스펙트럼에서 배음 구조를 유추할 수 있도록 도와주는 분석
  + 배음 구조 : 악기나 사람의 성도에 따라서 달라진다. 
  + 배음 : 한 음을 눌렀을 때 같이 울리는 파생음
  + 배음 구조의 차이가 악기의 소리, 목소리의 차이를 판단할 수 있다.
  + 켑스트럴 분석을 통해서 소리의 특징을 찾을 수 있다.
* 오디오 신호를 잘게 나눔 > 스펙트럼을 생성 > 멜 필터링 > 켑스트럴 분석(로그 → 역푸리에 변환)    
![이미지 009](https://user-images.githubusercontent.com/70581043/113291833-9014e800-932e-11eb-998f-9aeb1fb8a526.png)
* x축 : 시간, y축 : 주파수, z축 : 개숫값?
* 장점 : 역푸리에 변환을 통해서 주파수 정보의 상관 관계가 높은 문제를 해소  
![이미지 010](https://user-images.githubusercontent.com/70581043/113291860-9d31d700-932e-11eb-982c-20819f7fc226.png)




# 소리 데이터에 적합한 데이터 증강 기법 및 딥러닝 모델
### 소리에 적합한 Data Augmentation
1. Adding noise : 원본 오디오에 백색소음을 생성함
2. Shifting : 원본 오디오 데이터를 좌우로 이동함
3. Stretching : 원본 오디오 데이터의 빠르기 조정

### 소리 데이터를 위한 딥러닝 모델
* 구글 딥마인드 : wavenet 모델 공개
* TTS :  텍스트를 음성으로 변환

### Wavenet
* 오디오의 파형 형태를 직접 사용해서 새로운 파형을 생성하는 확률론적 모델
* 특징 3가지
  + 음성 파형 학습을 위한 새로운 딥러닝 구조를 제시함
  + 조건부 모델링을 이용해 특징적인 음성을 생성할 수 있음
  + 오디오 파형만을 이용해 자연스럽고 새로운 음성 파형을 생성할 수 있음


### SoftMax 함수
* 오디오 파일은 16비트의 정수 값으로 저장해서 사용함 >. 총 -215~215+1 (65536개) 사이의 숫자가 나옴 > 모델링하기 힘들다.
* u-law companding 변환을 통해 범위를 -127 ~ 128 사이의 정수(256개)로 바꾼다. > 총 256 개의 확률을 고려하면 됨

### Dilated Causal Convolutions
* Dilated convolution + Causal convolution 
* Dilated convolution : 파랑(input) -> 초록(output) 나머지는 0으로 채워진다.    
![이미지 011](https://user-images.githubusercontent.com/70581043/113291903-ac188980-932e-11eb-9d30-5eaaadef3486.png) 
  + 필터에 zero padding 을 추가해 모델의 receptive field를 늘려준다.
  + receptive field : 필터가 한 번에 볼 수 있는 데이터의 탐색 영역
  + 입련된 데이터의 특징을 잡아내기 위해서는 receptive field가 높으면 높을 수록 좋다.
  + 단점 : 연산이 너무 많아진다. 오버피팅 
* Causal convolution : 시간 순서를 고려하여 필터를 적용하는 컨볼루션 연산
  + 이전 레이어의 결과를 시프팅하면서 Conv1D 레이어에 쌓는다. 
  + RNN 계열의 모델처럼 시계열 데이터를 모델링 할 수 있음
  + 단점 : Receptive Field를 넓히기 위해서 많은 양의 레이어를 쌓아야 한다.    
  ![이미지 012](https://user-images.githubusercontent.com/70581043/113291943-baff3c00-932e-11eb-8e4a-ec9274c307bd.png)

* Dilated Causal convolution 
  + dilation 기법을 사용해 일정 스텝을 뛰면서 causal convolution 을 적용
  + 적게 레이어를 쌓아도 넓은 수용범위를 가질 수 있다.    
  ![이미지 013](https://user-images.githubusercontent.com/70581043/113291966-c6526780-932e-11eb-8b74-a007176b1d31.png)



### Conditional wavenet
* 확률 모델에 조건 정보를 추가함으로써 특정한 성질을 가진 오디오를 생성할 수 있음  
  + 전역적 조건 : 시점 변화에 영향을 받지 않는 정보를 추가, 본인이 갖고 있는 음색, 주파수는 변하지 않는다. 
  + 지역적 조건 : 시점 변화에 영향을 받는 정보를 추가, 특정 텍스트에 맞는 음성을 생성 (ex. '아이고'의 말투가 다르다.)


### Deep Learning model
* Feature - Mel spectrogram, MFCC를 사용할 것을 권장
* CNN Model - ResNet 모델을 사용할 것을 권장 

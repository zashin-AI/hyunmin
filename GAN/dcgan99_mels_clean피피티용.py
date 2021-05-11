from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf

#이미지가 저장될 폴더가 없다면 만듭니다.
# import os
# if not os.path.exists("E:\\nmb\\gan_images"):
#     os.makedirs("E:\\nmb\\gan_image")

np.random.seed(3)
tf.random.set_seed(3)

nosie_size = 100
x_size = 32
y_size = 128
gx1 = int(x_size/4)
gy1 = int(y_size/4)

epochs = 300001
batch = 16

#생성자 모델을 만듭니다.
# generator = Sequential()
# generator.add(Dense(128*gx1*gy1, input_dim=nosie_size, activation=LeakyReLU(0.2)))
# generator.add(BatchNormalization())
# generator.add(Reshape((gx1, gy1, 128)))
# generator.add(UpSampling2D())
# generator.add(Conv2D(64, kernel_size=5, padding='same'))
# generator.add(BatchNormalization())
# generator.add(Activation(LeakyReLU(0.2)))
# generator.add(UpSampling2D())
# generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))


generator = Sequential([
    Dense(128*gx1*gy1, 
          input_dim=nosie_size, 
          activation=LeakyReLU(0.2)),
    BatchNormalization(),
    Reshape((gx1, gy1, 128)),
    UpSampling2D(),
    Conv2D(64, kernel_size=5, 
           padding='same'),
    BatchNormalization(),
    Activation(LeakyReLU(0.2)),
    UpSampling2D(),
    Conv2D(1, kernel_size=5, 
           padding='same', 
           activation='tanh')
])

generator.summary()

# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_2 (Dense)              (None, 32768)             3309568   
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 32768)             131072    
# _________________________________________________________________
# reshape_1 (Reshape)          (None, 8, 32, 128)        0         
# _________________________________________________________________
# up_sampling2d_2 (UpSampling2 (None, 16, 64, 128)       0         
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 16, 64, 64)        204864    
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 16, 64, 64)        256       
# _________________________________________________________________
# activation_3 (Activation)    (None, 16, 64, 64)        0         
# _________________________________________________________________
# up_sampling2d_3 (UpSampling2 (None, 32, 128, 64)       0         
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 32, 128, 1)        1601      
# =================================================================
# Total params: 3,647,361
# Trainable params: 3,581,697
# Non-trainable params: 65,664

#판별자 모델을 만듭니다.
# discriminator = Sequential()
# discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(x_size,y_size,1), padding="same"))
# discriminator.add(Activation(LeakyReLU(0.2)))
# discriminator.add(Dropout(0.3))
# discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
# discriminator.add(Activation(LeakyReLU(0.2)))
# discriminator.add(Dropout(0.3))
# discriminator.add(Flatten())
# discriminator.add(Dense(1, activation='sigmoid'))

discriminator = Sequential([
    Conv2D(64, input_shape=(x_size,y_size,1), 
            strides=2, 
            kernel_size=5,
            padding="same"),
    Activation(LeakyReLU(0.2)),
    Dropout(0.3),
    Conv2D(128, kernel_size=5, 
            strides=2, 
            padding="same"),
    Activation(LeakyReLU(0.2)),
    Dropout(0.3),
    Flatten(),
    Dense(1, activation='sigmoid')
])

discriminator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_6 (Conv2D)            (None, 16, 64, 64)        1664      
# _________________________________________________________________
# activation_4 (Activation)    (None, 16, 64, 64)        0         
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 16, 64, 64)        0         
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 8, 32, 128)        204928    
# _________________________________________________________________
# activation_5 (Activation)    (None, 8, 32, 128)        0         
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 8, 32, 128)        0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 32768)             0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 32769     
# =================================================================
# Total params: 239,361
# Trainable params: 239,361
# Non-trainable params: 0
# _________________________________________________________________

#생성자와 판별자 모델을 연결시키는 gan 모델을 만듭니다.

discriminator.trainable = False
ginput = Input(shape=(nosie_size,))                  
x = generator(ginput)
dis_output = discriminator(x)   

# gan 모델 정의
gan = Model(ginput, dis_output)

gan.compile(loss='binary_crossentropy', optimizer='adam')

gan.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_2 (InputLayer)         [(None, 100)]             0         
# _________________________________________________________________
# sequential_2 (Sequential)    (None, 32, 128, 1)        3647361   
# _________________________________________________________________
# sequential_3 (Sequential)    (None, 1)                 239361    
# =================================================================
# Total params: 3,886,722
# Trainable params: 3,581,697
# Non-trainable params: 305,025
# _________________________________________________________________

X_train = np.load('E:\\nmb\\nmb_data\\npy\\2m5s\\project_total_npy\\total_f_data.npy')
print(X_train.shape)    # (2331, 128, 862)

x_train= []
for i in range (X_train.shape[0]) : 
    print(X_train[i].shape)
    img_resize = cv2.resize(X_train[i], (x_size, y_size))
    print(img_resize.shape)
    x_train.append(img_resize)
    
x_train = np.array(x_train)
print(x_train.shape)    # (2331, 128, 32)


from datetime import datetime
start = datetime.now()

#신경망을 실행시키는 함수를 만듭니다.
def gan_train(epoch, batch_size, saving_interval):
  
  global nosie_size, x_size, y_size


  # 데이터 불러오기
  print(x_train.shape)  # (2331, 128, 32)
  X_train = x_train.reshape(x_train.shape[0], x_size*y_size).astype('float32')
  
  print(np.max(X_train), np.min(X_train))   

  scale = MaxAbsScaler()
  X_train = scale.fit_transform(X_train)
  print(X_train.shape)  

  print(np.max(X_train), np.min(X_train))  
  
  X_train = X_train.reshape(x_train.shape[0], x_size, y_size, 1).astype('float32')

  print(X_train.shape)  # (2331, 32, 128, 1)

  true = np.ones((batch_size, 1))   # 1로 라벨링
  fake = np.zeros((batch_size, 1))  # 0으로 라벨링


  d_loss_list = list()
  g_loss_list = list()

  for i in range(epoch):

        idx = np.random.randint(0, X_train.shape[0], batch_size)    
        imgs = X_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true)

        noise = np.random.normal(0, 1, (batch_size, nosie_size))   
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        # GAN 훈련
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = gan.train_on_batch(noise, true)       

        print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)

        # loss 값 저장하기
        d_loss_list.append(float(d_loss))
        g_loss_list.append(float(g_loss))

        # pandas dataframe
        d_loss_df = pd.DataFrame(d_loss_list)
        g_loss_df = pd.DataFrame(g_loss_list)

        d_loss_df.columns = ['d_loss']
        g_loss_df.columns = ['g_loss']

        gd_loss_df = pd.concat([d_loss_df, g_loss_df], axis = 1)

        gd_loss_df.to_csv('E:\\nmb\\gan_image\\csv\\loss_dcgan15.csv', index = False)

        if i % saving_interval == 0:
            # saving_interval 마다 이미지를 출력한다.
            noise = np.random.normal(0, 1, (25, nosie_size))
                # print(noise.shape)  # (25, 100)
            gen_imgs = generator.predict(noise)
                # print(gen_imgs.shape)   # (25, 28, 28, 1)
            # idx2 = np.random.randint(0, gen_imgs.shape[0])
            idx2 = 5    # 출력 값 고정
            print("id2 > ", idx2)
            
            gen_imgs = 0.5 * gen_imgs + 0.5
                # Rescale images 0 - 1
            print(gen_imgs.shape) # (25, 32, 128, 1)
       
            # 다시 melspectrogram으로 바꾸기 (shape 맞추기)
            gen_mels = cv2.resize(gen_imgs[idx2], (862,128))
            # gen_mels = cv2.resize(gen_imgs[idx2], (128, 862))
            # print(gen_mels.shape) # (862, 128)
            # gen_mels = gen_mels.reshape(gen_mels.shape[1], gen_mels.shape[0] )
            print(gen_mels.shape) # (128, 862)

            # 다시 melspectrogram으로 바꾸기
            remels = librosa.feature.inverse.mel_to_audio (gen_mels, sr=22050, hop_length=128, n_fft=512)
            print(remels.shape)   # (110208,)

            sf.write(f'E:\\nmb\\gan_image\\dcgan15\\{i}.wav', remels, 22050)

            # librosa.display.waveplot(remels, 22050)
            # plt.show()
            print("end")
            

gan_train(epochs, batch, 500)


end = datetime.now()
time = end - start
print("작업 시간 : " , time)  
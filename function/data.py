import datasave
import numpy as np

filepath = 'E:/nmb/nmb_data/ForM/M/' # 파일 경로
filename = 'flac'                    # 파일 확장자
labels = 1                           # 라벨값 지정

dataset, label = datasave.load_data_mel(filepath, filename, labels)
dataset = np.array(dataset)
label = np.array(label)

print(dataset.shape)    # (528, 128, 862)
print(label.shape)      # (528,)

print("==============")

filepath = 'E:/nmb/nmb_data/ForM/M/' # 파일 경로
filename = 'flac'                    # 파일 확장자
labels = 1                           # 라벨값 지정

dataset, label = datasave.load_data_mfcc(filepath, filename, labels)
dataset = np.array(dataset)
label = np.array(label)

print(dataset.shape)    # (528, 20, 862)
print(label.shape)      # (528,)


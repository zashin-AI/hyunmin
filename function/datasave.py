import librosa
import numpy as np
import sklearn

def load_data_mfcc(filepath, filename, labels):

    count = 1
    dataset = list()
    label = list()
    
    def normalize(x, axis = 0):
        return sklearn.preprocessing.minmax_scale(x, axis = axis)

    files = librosa.util.find_files(filepath, ext=[filename])
    files = np.asarray(files)
    for file in files:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mels = librosa.feature.mfcc(y, sr=sr, n_fft=512, hop_length=128, n_mfcc=20)
            mels = librosa.amplitude_to_db(mels, ref=np.max)
            mels = normalize(mels, axis = 1)

            dataset.append(mels)
            label.append(labels)
            
            count+=1

    return dataset, label

def load_data_mel(filepath, filename, labels):

    '''
    Args :
        filepath : 파일 불러올 경로
        filename : 불러올 파일 확장자명 e.g. wav, flac...
        label : label 번호 (여자 : 0, 남자 : 1)
    '''
    count = 1

    dataset = list()
    label = list()

    files = librosa.util.find_files(filepath, ext=[filename])
    files = np.asarray(files)
    for file in files:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
            mels = librosa.amplitude_to_db(mels, ref=np.max)

            dataset.append(mels)
            label.append(labels)
            
            count+=1

    return dataset, label


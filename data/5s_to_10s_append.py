from pydub import AudioSegment, effects
import librosa, sys, os
import numpy as np
sys.path.append('E:/nmb/nada/python_import/')
import copy


# folder_path = 'C:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\F\\'
folder_path = 'C:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\M\\'

for i in range(43) : 
    # out_file = folder_path + "F" + str(i+1) +".wav"    # wav 파일 생성하고 작업 끝나면 지우기
    out_file = folder_path + "M" + str(i+1) +".wav"    # wav 파일 생성하고 작업 끝나면 지우기
    print(out_file)
    audio_copy = AudioSegment.from_wav(out_file)

    audio_copy = copy.deepcopy(audio_copy)
    for num in range(1) :
        audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0) 
    # f_10s_file = folder_path + "10s_F\\"+  "F" + str(i+1) + "_10s.wav"
    f_10s_file = folder_path + "10s_M\\"+  "M" + str(i+1) + "_10s.wav"
    audio_copy.export(f_10s_file , format='wav')


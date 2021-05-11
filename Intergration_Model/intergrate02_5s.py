from pydub import AudioSegment, effects
from pydub.silence import split_on_silence, detect_silence      
import speech_recognition as sr
from hanspell import spell_checker     
import librosa.display
import librosa, sys, os
import numpy as np
import noisereduce as nr
import soundfile as sf
sys.path.append('E:/nmb/nada/python_import/')
import copy
from tensorflow.keras.models import load_model

'''
[순서]
남성여성 대화형 음성 데이터
> 디노이즈 
> 볼륨 정규화 
> 묵음 부분마다 음성 자름 
> google stt에 적용 
> 한글 맞춤법 검사 
> 화자 구분 
> 결과 출력
'''

# 남녀가 말하는 음성 파일 입력 
audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\pansori\\un4qbATrmx8.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\mindslabABS\\ABS_M_81_SE_2018-0808-1145-40_denoise.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\mindslabGYD\\GYD_M_88_DG_2018-0806-1105-38_denoise.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\korea_t12\\korea_multi_t12.wav'
# audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\korea_t18\\korea_multi_t18.wav'

# 파일 경로 분리
audio_file_path = os.path.splitext(audio_file)
audio_file_path = os.path.split(audio_file_path[0])
folder_path = audio_file_path[0]
file_name = audio_file_path[1]
print(folder_path, file_name)

# class intergrate_model : 

def _denoise (audio_file) : 
    '''
    노이즈 제거
    '''
    data, samplig_rate = librosa.load(audio_file) 
    noise_part = data[5000:15000]
    reduce_noise = nr.reduce_noise(
        audio_clip=data, 
        noise_clip=noise_part,
        n_fft = 512,
        hop_length = 128,
        win_length = 512
    )
    sf.write(audio_file, data, samplig_rate)

def _normalized_sound(audio_file) : 
    '''
    볼륨 정규화
    '''
    audio = AudioSegment.from_wav(audio_file)
    normalizedsound = effects.normalize(audio)
    return normalizedsound

def _split_silence(audio_file) :
    '''
    묵음마다 음성 자르기
    '''
    dbfs = audio_file.dBFS
    audio_chunks = split_on_silence(
        audio_file,  
        min_silence_len= 500,
        silence_thresh= dbfs - 16,
        keep_silence= 300
    )
    return audio_chunks

def _STT_checked_hanspell (audio_file) :
    '''
    STT & 한글 맞춤법 확인
    '''
    txt = r.recognize_google(audio_file, language="ko-KR")
    spelled_sent = spell_checker.check(txt)
    checked_sent = spelled_sent.checked
    return checked_sent  

def _predict_speaker(y, sr) :
    '''
    여자(0) , 남자(1) 예측하기
    '''
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    # print(pred_mels.shape)  # (1, 128, 862)
    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :                   
        # print('여자')
        return '여자'
    else:                               
        # print('남자')  
        return '남자'  
    
# 디노이즈
# _denoise(audio_file)
# print("denoise done")

# 볼륨 정규화
normalizedsound = _normalized_sound(audio_file)
print("normalized done")

# 묵음 자르기
audio_chunks = _split_silence(normalizedsound)
# print(audio_chunks)
# [<pydub.audio_segment.AudioSegment object at 0x000001CF0F383C88>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F383CC0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F383D68>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5A90>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5B00>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5B38>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5BE0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F7790F0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB908>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB9E8>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB390>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB3C8>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB400>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB438>]
# print("len(audio_chunks)", len(audio_chunks))    # 14
len_audio_chunks = len(audio_chunks)

# 화자 구분을 가장 잘하는 모델 load
model = load_model('E:/nmb/nmb_data/cp/mobilenet_rmsprop_1.h5')

r = sr.Recognizer()
save_script = ''

# STT -> 화자구분
for i, chunk in enumerate(audio_chunks): 
    speaker_stt = []   
    out_file = folder_path + "\\"+ str(i) + "_chunk.wav"    # wav 파일 생성 안하고 STT로 바꿀 수 있는 방법은 없을까//?
    # chunk.export(out_file, format="wav")
    aaa = sr.AudioFile(out_file)
    with aaa as source :
        audio = r.record(aaa)

    try : 
        # [1] STT & 맞춤법 확인
        spell_checked_text = _STT_checked_hanspell(audio)
        speaker_stt.append(str(spell_checked_text))     # 화자와 텍스트를 한 리스트로 합칠 것임

        # [2] 화자구분
        y, sampling_rate = librosa.load(out_file, sr=22050)

        if len(y) >= 22050*5 : # 5초 이상이라면,
            y = y[:22050 * 5]  # 5초만 model.predict에 사용할 것임
            speaker = _predict_speaker(y, sampling_rate)
            speaker_stt.append(str(speaker))
            print(speaker_stt[1], " : " , speaker_stt[0])

        else :  # 5초 미만인 파일을 5초 이상으로 만든 후, 5초로 잘라서 model.predict에 넣는다.
            audio_copy = AudioSegment.from_wav(out_file)
            audio_copy = copy.deepcopy(audio_copy)
            for num in range(3) :
                audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0) 
            audio_copy.export(folder_path + "\\"+ str(i) + "_chunk_over_5s.wav", format='wav')
            y_copy, sampling_rate = librosa.load(folder_path + "\\"+ str(i) + "_chunk_over_5s.wav", sr=22050)
            y_copy = y_copy[:22050 * 5]
            speaker = _predict_speaker(y_copy, sampling_rate)
            speaker_stt.append(str(speaker))    # 화자 구분을 못했다는 걸 공백으로 저장
            print(speaker_stt[1], " : " , speaker_stt[0])
        
        # txt 파일로 저장하기
        save_script += speaker_stt[1] +': ' + speaker_stt[0] + '\n\n'
        with open(folder_path + "\\stt_script_5s.txt", 'wt') as f: f.writelines(save_script) 

    except : 
        # 너무 짧은 음성은 STT & 화자구분 pass 
        pass   

"""
[문제점]
- 5초로 model.predict로 만들어서, 5초 미만은 화자구분을 못한다. 화자 구분 안하고 넘어가는 게 많다.
    > 1초로 model 만들어야 할 듯 ? ^ ^ ^^ 
    >> 아니? 그럴 수 없어 
    >> 일부러 5초 이상 만들어서 5초로 잘라서 넣는다.
- 묵음 제거한 후, 같은 음성 파일 내에 여자 남자 동시에 있는 경우들이 있다. 
    > 묵음 제거를 잘 해야 함

[개선해야 할 점]
- class로 만들어보자 ~
- 묵음 제거 한 다음에 wav 파일 저장하는 과정없이 바로 stt랑 화자구분할 수는 없을까? 
- 화자 구분 더 잘하는 모델있으면 그걸로 가중치 넣기
- 묵음 제거 개선
- STT 개선
"""


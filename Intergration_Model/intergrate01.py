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

from tensorflow.keras.models import load_model

'''
2분 남성여성 대화형 음성 데이터
> 디노이즈 
> 볼륨 정규화 
> 묵음 부분마다 음성 자름 
> google stt에 적용 
> 한글 맞춤법 검사 
> 화자 구분 
> 결과 출력
'''

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
    sf.write(audio_file, reduce_noise, samplig_rate)
    return reduce_noise

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
    

audio_file = 'E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\un4qbATrmx8.wav'

# 디노이즈
# _denoise(audio_file)
# print("denoise done")

# 볼륨 정규화
normalizedsound = _normalized_sound(audio_file)
print("normalized done")

# 묵음 자르기
audio_chunks = _split_silence(normalizedsound)
print(audio_chunks)
# [<pydub.audio_segment.AudioSegment object at 0x000001CF0F383C88>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F383CC0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F383D68>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5A90>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5B00>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5B38>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F6C5BE0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0F7790F0>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB908>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB9E8>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB390>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB3C8>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB400>, <pydub.audio_segment.AudioSegment object at 0x000001CF0FCCB438>]
print("len(audio_chunks)", len(audio_chunks))    # 14
len_audio_chunks = len(audio_chunks)

# 화자 구분을 가장 잘하는 모델 load
model = load_model('E:/nmb/nmb_data/cp/conv2d/Conv2D_model_Adadelta000001_16.h5')

r = sr.Recognizer()
result = [[] for i in range(len_audio_chunks)]

# STT -> 화자구분
for i, chunk in enumerate(audio_chunks): 
    speaker_stt = []   
    out_file = "E:\\nmb\\nmb_data\\STT_multiple_speaker_temp\\"+ str(i) + f"_chunk{i}.wav"    # wav 파일 생성 안하고 STT로 바꿀 수 있는 방법은 없을까//?
    # chunk.export(out_file, format="wav")
    aaa = sr.AudioFile(out_file)
    with aaa as source :
        audio = r.record(aaa)
    # print(audio)

    try : 
        # [1] STT & 맞춤법 확인
        spell_checked_text = _STT_checked_hanspell(audio)
        speaker_stt.append(str(spell_checked_text))     # 화자와 텍스트를 한 리스트로 합칠 것임
        # print(spell_checked_text)

        # [2] 화자구분
        y, sampling_rate = librosa.load(out_file, sr=22050)
        # print(len(y), sampling_rate)

        if len(y) >= 22050*5 : # 5초 이상이라면,
            y = y[:22050 * 5]  # 5초만 model.predict에 사용할 것임
            speaker = _predict_speaker(y, sampling_rate)
            speaker_stt.append(str(speaker))
            # print(len(speaker_stt))

            print(speaker_stt[1], " : " , speaker_stt[0])
            # result[i].append(speaker_stt)

        else :
            # 5초 미만인 파일은 model.predict 못함
            speaker_stt.append(' ')    # 화자 구분을 못했다는 걸 공백으로 저장
            print(speaker_stt[0])
        
        result[i] = speaker_stt
        
    except : 
        # 너무 짧은 음성은 STT & 화자구분 pass 
        pass   

print(result)
print(result[0][1], " : " , result[0][0])   # 화자가 없을 때 >    :  저희는 방금 소개받은 것처럼 의사고요
print(result[1][1], " : " , result[1][0])   # 화자가 있을 때 > 여자  :  여기가 저희 진료실입니다 저는 이렇게 많아 책을 보고 있었고 전시를 하기도 합니다 뭘 하고 있거든요 근데 이게 뭘 하는 게 저희가 노는 그런 게 아니라 진짜 뭔가 저희가 꿈꾸는 뭔가가 있기 때문에 하는 건데 것이 바로
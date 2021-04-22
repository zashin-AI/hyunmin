# https://github.com/jiaaro/pydub/issues/169
import sys
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa
sys.path.append('E:/nmb/nada/python_import/')
from voice_handling import import_test, voice_sum


r = sr.Recognizer()


'''디노이즈 > 속도 느리게 > 오디오 파일을 불러옴 > silence 부분마다 잘라서 음성 파일 저장 > 해당 파일을 google stt에 적용 > 한글 맞춤법 검사 > text 출력'''

"""
file_list = librosa.util.find_files('E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_2m', ext=['wav'])
print(file_list)

for j, path in enumerate(file_list) : 
    # 오디오 불러오기
    sound_file = AudioSegment.from_wav(path)
    # 가장 최소의 dbfs가 무엇인지
    # dbfs : 아날로그 db과는 다른 디지털에서의 db 단위, 0일 때가 최고 높은 레벨
    dbfs = sound_file.dBFS
    # print(sound_file.dBFS)
    # silence 부분 마다 자른다. 
    audio_chunks = split_on_silence(sound_file,  
        min_silence_len= 200,
        silence_thresh= dbfs - 16 ,
        # keep_silence= 100
        keep_silence= 0
    )
    # print(len(audio_chunks))
    full_txt = []
    # 말 자른 거 저장
    for i, chunk in enumerate(audio_chunks):        
        out_file = "E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\"+ f"m_chunk{i}.wav"
        print ("exporting", out_file)
        # chunk.export(out_file, format="wav")


path_wav = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\'
path_out = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\total\\mindslab_m_silence_total.wav'
voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)
"""

path = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_2m\\m1.wav'

# 오디오 불러오기
sound_file = AudioSegment.from_wav(path)
# 가장 최소의 dbfs가 무엇인지
# dbfs : 아날로그 db과는 다른 디지털에서의 db 단위, 0일 때가 최고 높은 레벨
dbfs = sound_file.dBFS
# print(sound_file.dBFS)
# silence 부분 마다 자른다. 
audio_chunks = split_on_silence(sound_file,  
    min_silence_len= 200,
    silence_thresh= dbfs - 16 ,
    # keep_silence= 100
    keep_silence= 0
)
# print(len(audio_chunks))
full_txt = []
# 말 자른 거 저장
for i, chunk in enumerate(audio_chunks):        
    out_file = "E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\m1_chunk\\"+ f"m1_chunk{i}.wav"
    print ("exporting", out_file)
    chunk.export(out_file, format="wav")

 
path_wav = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\m1_chunk\\'
path_out = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\total\\mindslab_m1_silence_total.wav'
voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)

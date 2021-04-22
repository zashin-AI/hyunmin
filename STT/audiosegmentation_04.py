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

def split_silence_hm (audio_dir, split_silence_dir, sum_dir) :

    file_num = len(audio_dir)
    # print(file_num)

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    for path in audio_dir :
        print(path)

        # 오디오 불러오기
        sound_file = AudioSegment.from_wav(path)

        # audio = AudioSegment.from_file(origin_dir)
        _, w_id = os.path.split(path)
        w_id = w_id[:-4]

        # 가장 최소의 dbfs가 무엇인지
        # dbfs : 아날로그 db과는 다른 디지털에서의 db 단위, 0일 때가 최고 높은 레벨
        dbfs = sound_file.dBFS

        # silence 부분 마다 자른다. 
        audio_chunks = split_on_silence(sound_file,  
            min_silence_len= 200,
            silence_thresh= dbfs - 16 ,
            # keep_silence= 100
            keep_silence= 0
        )

        createFolder( split_silence_dir + w_id )

        # 말 자른 거 저장
        for i, chunk in enumerate(audio_chunks):        
            out_file = split_silence_dir + w_id + "\\" + w_id+ f"_{i}.wav"
            # print ("exporting", out_file)
            chunk.export(out_file, format="wav")

        ###############################################

        # [2] 묵음을 기준으로 자른 오디오 파일을 하나의 파일로 합친다.
        path_wav = split_silence_dir + w_id + "\\" 
        print(path_wav) # E:\nmb\nmb_data\mindslab\minslab_m\m_total_chunk\f7
        path_out = sum_dir + w_id + '_silence_total.wav'
        print(path_out) # E:\nmb\nmb_data\mindslab\minslab_m\m_total_chunk\total\f7_silence_total.wav
        voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)

audio_dir = librosa.util.find_files('E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_2m\\', ext=['wav'])
split_silence_dir = "E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\"
sum_dir = "E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\total\\"

split_silence_hm(audio_dir, split_silence_dir, sum_dir)

######################################################################################
import sys
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence   
import librosa
from voice_handling import import_test, voice_sum

# split_silence_hm : 오디오 파일 침묵구간 마다 오디오 자름 > 자른 오디오 저장 > 합친 오디오 저장
# audio_dir (여러 오디오가 있는 파일경로)
# split_silence_dir (묵음 부분 마다 자른 오디오 파일을 저장할 경로)
# sum_dir (wav 파일을 합쳐서 저장할 파일경로)

def split_silence_hm(audio_dir, split_silence_dir, sum_dir) :
    
    audio_dir = librosa.util.find_files(audio_dir, ext=['wav'])                             # audio_dir에 있는 모든 파일을 가져온다.

    for path in audio_dir :                                                                 # audio_dir에 있는 파일을 하나 씩 불러온다.

        sound_file = AudioSegment.from_wav(path)
        _, w_id = os.path.split(path)
        w_id = w_id[:-4]

        dbfs = sound_file.dBFS                                                                
        audio_chunks = split_on_silence(
            sound_file,                                                                     # silence 부분 마다 자른다.
            silence_thresh= dbfs - 16 ,
            min_silence_len= 200,
            keep_silence= 0
        )
                                                                       
        for i, chunk in enumerate(audio_chunks):                                            # silence 부분 마다 자른 거 wav로 저장
            out_file = split_silence_dir + w_id + "\\" + w_id+ f"_{i}.wav"
            chunk.export(out_file, format="wav")
                                                                                            
        path_wav = split_silence_dir + w_id + "\\"                                          # 묵음을 기준으로 자른 오디오 파일을 하나의 파일로 합친다. # 묵음으로 잘린 파일이 저장된 곳
        path_out = sum_dir + w_id + '_silence_total.wav'                                    # 오디오 합친 파일 경로
        voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)

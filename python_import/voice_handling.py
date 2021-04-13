# 오디오 합치기와 나누기를 함수로 정의해보자!

import librosa
from pydub import AudioSegment
import soundfile as sf
import os

# 테스트 한번 불러서 출력하고 가져가세요~
def import_test():
    print('==== it will be great ====')

# ---------------------------------------------------------------
# voice_sum: 오디오 한 wav 파일로 합쳐서 저장하기

# example
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
# out_dir(wav파일을 합쳐서 저장할 경로+파일명) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

def voice_sum(form, audio_dir, save_dir, out_dir):
    if form =='flac':
        infiles = librosa.util.find_files(audio_dir)
        for infile in infiles:
            # flac 파일의 이름 불러오기
            _, w_id = os.path.split(infile)
            # 이름에서 뒤에 .flac 떼기
            w_id = w_id[:-5]
            # flac 파일의 data, sr 불러오기
            w_data, w_sr = sf.read(infile)
            # 같은 이름으로 wav 형식으로 저장하기
            sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
        print('==== flac to wav done ====')
        # 모든 파일 불러오기
        infiles = librosa.util.find_files(save_dir)
        # 모든 경로를 wav 불러오기로
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        # 맨 처음 시작 지정
        combined = wavs[0]
        # 1번부터 이어 붙이기
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        # out_dir 경로에 wav 형식으로 내보내기
        combined.export(out_dir, format='wav')
        print('==== wav sum done ====')

    if form == 'wav':
        infiles = librosa.util.find_files(audio_dir)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        combined.export(out_dir, format='wav')
        print('==== wav sum done ====')


# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기

# **** example ****
# origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'

def voice_split(origin_dir, threshold, out_dir):
    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    lengaudio = len(audio)
    # 임계점 설정(1s = 1000ms)
    start = 0
    threshold = threshold
    end = 0
    counter = 0
    # 본격적인 잘라서 저장하기
    while start < len(audio):
        end += threshold
        print(start, end)
        chunk = audio[start:end]
        filename = out_dir + w_id + f'{counter}.wav'
        chunk.export(filename, format='wav')
        counter += 1
        start += threshold
    print('==== wav split done ====')

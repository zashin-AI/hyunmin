import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_split, voice_split_1m, voice_split_term

import_test()
# ==== it will be great ====

# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기
# def voice_split(origin_dir, threshold, out_dir):
# **** example ****
# origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'

# 적용해보자!
filename = 'audio_hgh'
origin_dir = 'E:\\nmb\\nmb_data\\youtube\\'+ filename + '.wav'
threshold = 5000 # 몇초씩 자를 것인지 설정
out_dir = 'E:\\nmb\\nmb_data\\youtube\\5s\\'
end_threshold = 5000 # 끝나는 지점(1분)
start = 113000
end = start + 5000
voice_split_term(origin_dir=origin_dir, out_dir=out_dir, start=start, end=end)
# 잘 된다잉~


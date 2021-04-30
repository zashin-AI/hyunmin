import librosa
from pydub import AudioSegment
import soundfile as sf
import os
import sys
sys.path.append('E:/nmb/nada/python_import/')
from voice_handling import import_test, voice_split, voice_split_1m, voice_split_term
from noise_handling import denoise_tim

# ---------------------------------------------------------------
# voice_sum: 오디오 한 wav 파일로 합쳐서 저장하기
# def voice_sum(form, pathaudio, save_dir, out_dir):
# **** example ****
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
# out_dir(wav파일을 합쳐서 저장할 경로+파일명까지) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

# [1] 문장별로 잘려져 있는 음성 파일 하나로 합치기
# # 2) flac일 때
# path_flac = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\un4qbATrmx8\\'
# path_save = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\un4qbATrmx8_wav\\'
# path_out = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\un4qbATrmx8.wav'
# voice_sum(form='flac', audio_dir=path_flac, save_dir=path_save, out_dir=path_out)


# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기
# def voice_split(origin_dir, threshold, out_dir):
# **** example ****
# origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'

# [2] 음성 파일 5분 미만으로 맞추기
# name = ['ABS_M_81_SE_2018-0808-1145-40','GYD_M_88_DG_2018-0806-1105-38','KBH_M_84_DG_2018-0807-1007-39','LSM_M_81_WS_2018-0814-1037-53','un4qbATrmx8']
# for filename in name :
#     origin_dir = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\wav\\'+ filename + '.wav'
#     # threshold = 0 # 몇초씩 자를 것인지 설정
#     out_dir = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\'
#     # end_threshold = 5000 # 끝나는 지점(1분)
#     start = 0
#     end = start + 300000
#     voice_split_term(origin_dir=origin_dir, out_dir=out_dir, start=start, end=end)


# [3] 디노이즈 적용하기
denoise_tim(
    load_dir = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\5m_wav',
    out_dir = 'E:\\nmb\\nmb_data\\5m_fm_dialog\\',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)

import os
import sys
sys.path.append('E:/nmb/nada/python_import/')

from noise_handling import denoise_tim

denoise_tim(
    load_dir = 'E:/nmb/nmb_data/audio_data/',
    out_dir = 'E:/nmb/nmb_data/audio_data_noise/',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)
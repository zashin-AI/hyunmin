import os
import sys
sys.path.append('E:/nmb/nada/python_import/')
from noise_handling import denoise_tim

# denoise_tim(
#     load_dir = 'E:/nmb/nmb_data/audio_data/',
#     out_dir = 'E:/nmb/nmb_data/audio_data_noise/',
#     noise_min = 5000,
#     noise_max = 15000,
#     n_fft = 512,
#     hop_length = 128,
#     win_length = 512
# )

denoise_tim(
    # load_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\mindslab_m_total\\',
    # out_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\',
    load_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\mindslab_f_total\\',
    out_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\',
    noise_min = 5000,
    noise_max = 15000,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)

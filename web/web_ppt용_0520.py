from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
from hanspell import spell_checker  

import numpy as np
import librosa
import speech_recognition as sr
import tensorflow as tf
import os
import copy


        
r = sr.Recognizer()

def normalized_sound(auido_file):
    audio = AudioSegment.from_wav(auido_file)
    normalizedsound = effects.normalize(audio)
    return normalizedsound

def split_slience(audio_file):
    dbfs = audio_file.dBFS
    audio_chunks = split_on_silence(
        audio_file,
        min_silence_len=1000,
        silence_thresh=dbfs - 30,
        keep_silence=True
    )
    return audio_chunks

def STT_hanspell(audio_file):
    with audio_file as audio:
        file = r.record(audio)
        stt = r.recognize_google(file, language='ko-KR')
        spelled_sent = spell_checker.check(stt)
        checked_sent = spelled_sent.checked
    return checked_sent

def predict_speaker(y, sr):
    mels = librosa.feature.melspectrogram(y, sr = sr, hop_length=128, n_fft=512, win_length=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])

    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0:
        return '여자'
    if y_pred_label == 1:
        return '남자'

app = Flask(__name__)

@app.route('/')
def upload_file() :
    return render_template('upload.html')

@app.route('/uploadFile', methods = ['POST'])
def download() :
    if request.method == 'POST' :
        f = request.files['file']
        if not f : return render_template('upload.html')

        folder_path = 'E:/nmb/nmb_data/web/chunk/'

        normalizedsound = normalized_sound(f)
        audio_chunks = split_slience(normalizedsound)
        save_script = ''

        for i, chunk in enumerate(audio_chunks):
            speaker_stt = list()
            out_file = folder_path + '/' + str(i) + '_chunk.wav'
            chunk.export(out_file, format = 'wav')
            aaa = sr.AudioFile(out_file)

            try :
                stt_text = STT_hanspell(aaa)
                speaker_stt.append(str(stt_text))
                y, sample_rate = librosa.load(out_file, sr = 22050)

                if len(y) >= 22050*5:
                    y = y[:22050*5]
                    speaker = predict_speaker(y, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1], " : ", speaker_stt[0])
                
                else:
                    audio_copy = AudioSegment.from_wav(out_file)
                    audio_copy = copy.deepcopy(audio_copy)
                    for num in range(3):
                        audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0)
                    audio_copy.export(folder_path + '/' + str(i) + '_chunks_over_5s.wav', format = 'wav')
                    y_copy, sample_rate = librosa.load(folder_path + '/' + str(i) + '_chunks_over_5s.wav')
                    y_copy = y_copy[:22050*5]
                    speaker = predict_speaker(y_copy, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1] + " : " + speaker_stt[0])
                
                save_script += speaker_stt[1] + " : " + speaker_stt[0] + '\n\n'
                with open('E:/nmb/nada/web/static/test.txt', 'wt', encoding='utf-8') as f: f.writelines(save_script)
        
            except : pass
    return render_template('/download.html')

# 파일 다운로드
@app.route('/downlaod/')
def download_file() :
    file_name = 'E:/nmb/nada/web/static/test.txt'
    return send_file(
        file_name,
        as_attachment=True,
        mimetype='text/txt',
        cache_timeout=0
    )

# 추론 된 파일 읽기
@app.route('/read/')
def read_text() :
    f = open('E:/nmb/nada/web/static/test.txt', 
              'r', encoding='utf-8')
    return "</br>".join(f.readlines())

if __name__ == "__main__" :
    model = load_model('E:/nmb/nmb_data/cp/mobilenet_rmsprop_1.h5')
    app.run(debug=True)


from pydub import AudioSegment
import soundfile as sf
import librosa.display
import librosa
import os

# https://www.javaer101.com/ko/article/33541788.html
def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
     # convert the sound with altered frame rate to a standard frame rate
     # so that regular playback programs will work right. They often only
     # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

"""
file_list = librosa.util.find_files('E:\\nmb\\nmb_data\\predict\\stt_denoise\\denoise\\', ext=['wav'])
for i, sound in enumerate(file_list) : 
    # print(sound)
    name = os.path.basename(sound)
    # print(name)
    sound = AudioSegment.from_file(sound)
    out_file = "E:\\nmb\\nmb_data\\predict\\stt_denoise\\"+ str(name) + "_slow.wav"

    slow_sound = speed_change(sound, 0.90)
    slow_sound.export(out_file, format="wav")

    # fast_sound = speed_change(sound, 2.0)
    # fast_sound.export("E:\\nmb\\nmb_data\\predict\\stt_denoise\\test_01_fast.wav", format="wav")
"""

sound = AudioSegment.from_file('E:\\nmb\\nmb_data\\predict\\M3.wav')
out_file = "E:\\nmb\\nmb_data\\predict\\M3_slow.wav"
slow_sound = speed_change(sound, 0.90)
slow_sound.export(out_file, format="wav")

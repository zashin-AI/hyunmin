from pytube import YouTube
import glob
import os.path

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
# par = 'https://www.youtube.com/watch?v=VlHev1GA10E'   # 황광희
# par = 'https://www.youtube.com/watch?v=ahl3ZeWPhgQ'   # 유재석
# yt = YouTube(par)
# yt.streams.filter()

# yt.streams.filter().first().download()
# print('success')

# 그 다음 실행 2번
import moviepy.editor as mp

output_path = 'E:\\nmb\\nmb_data\\youtube\\5s'
filename = '[2020 MBC 방송연예대상] 놀면 뭐하니 유재석 대상 수상!!! MBC 201229 방송'
clip = mp.VideoFileClip(filename + ".mp4")
clip.audio.write_audiofile(output_path + filename + '.wav')
clip.audio.write_audiofile("audio_yjs.wav")
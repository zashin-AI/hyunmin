# 유튜브에서 5초씩 음성 가져오기! 가져와서 자르기도 여기서 해야지~
# 종호가 만든 파일 참고함

from pytube import YouTube
import glob
import os.path

# --------------------------------------------------------------------------------------------
# 유튜브 저장하기
output_path = 'E:/nmb/nmb_data/youtube/'
filename = 'm16'

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
par = 'https://www.youtube.com/watch?v=rkO67Mvhu-A'
yt = YouTube(par)
yt.streams.filter()

output_path = 'E:/nmb/nmb_data/youtube/'
yt.streams.filter().first().download(output_path = output_path)
print('==== youtube audio save done ====')


# 그 다음 실행 2번
import moviepy.editor as mp

clip = mp.VideoFileClip(output_path + filename + '.mp4')
clip.audio.write_audiofile(output_path + filename + '.wav')
print('==== wav save done ====')



# --------------------------------------------------------------------------------------------
# 그 다음 실행 3번
# 5초 자르기
from voice_handling import voice_split_term

origin_dir = output_path+ filename + '.wav'
out_dir = output_path
start = ((2*60)+0)*1000
end = start + 5000
voice_split_term(origin_dir=origin_dir, out_dir=out_dir, start=start, end=end)
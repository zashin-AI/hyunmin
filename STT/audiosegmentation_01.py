# https://github.com/jiaaro/pydub/issues/169
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
r = sr.Recognizer()

'''오디오 파일을 불러옴 > silence 부분마다 잘라서 음성 파일 저장 > 해당 파일을 google stt에 적용 > text 출력'''

# 오디오 불러오기
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\youtube\\5s\\F6.wav")
    # 그냥 지켜보며
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\youtube\\5s\\F42.wav")
    # 동료들도
    # 저는 나무
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\youtube\\5s\\M14.wav")
    # 해야 될 거 아니야
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\youtube\\5s\\M15.wav")
    # 지금은
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\youtube\\5s\\M16.wav")
    # 출연자
    # 조용히 해
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\youtube\\5s\\M41.wav")   
    # 연기가
sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\predict\\M\\M5.wav") 
    # 거리두기 집
    # 자리는 비
# sound_file = AudioSegment.from_wav("E:\\nmb\\nmb_data\\predict\\M\\M11.wav")
    # 되게 좋았던
    # 이야기 전개

# 가장 최소의 dbfs가 무엇인지
dbfs = sound_file.dBFS
# print(sound_file.dBFS)
thresh = int(dbfs)
# print(int(sound_file.dBFS))

# 최소의 dbfs를 threshold에 넣는다.
if dbfs < thresh :
    thresh = thresh - 1
    # print(thresh)

# silence 부분 마다 자른다. 
audio_chunks = split_on_silence(sound_file,  
    # split on silences longer than 1000ms (1 sec)
    min_silence_len=500,

    # anything under -16 dBFS is considered silence
    silence_thresh= thresh , 

    # keep 200 ms of leading/trailing silence
    keep_silence=200
)

# 말 자른 거 저장 & STT 
for i, chunk in enumerate(audio_chunks):

    out_file = "E:\\nmb\\nmb_data//chunk{0}.wav".format(i)
    # print ("exporting", out_file)
    chunk.export(out_file, format="wav")
    aaa = sr.AudioFile(out_file)
    with aaa as source :
        audio = r.record(aaa)#, duration=1) 
    # print(type(audio))
    try:
        txt = r.recognize_google(audio, language="ko-KR")
        print(txt)
    except : # 너무 짧은 음성은 pass 됨 
        pass
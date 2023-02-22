# import moviepy.editor as mp
# import speech_recognition as sr
# import pysrt
# from pocketsphinx import LiveSpeech, get_model_path
# # set path to language model
# model_path = get_model_path()
# # video file path
# video_file = r'C:\Users\coool\Desktop\NLP project\Coldplay - Hymn For The Weekend (Official Video).mkv'
# # extract audio from video
# video = mp.VideoFileClip(video_file)
# audio = video.audio
# audio.write_audiofile('audio.wav')
# # transcribe audio to text
# r = sr.Recognizer()
# with sr.AudioFile('audio.wav') as source:
#     audio = r.record(source)
# speech = LiveSpeech(
#     verbose=False,
#     sampling_rate=16000,
#     buffer_size=2048,
#     no_search=False,
#     full_utt=False,
#     hmm=model_path,
#     lm=model_path + '/en-us.lm.bin',
#     dic=model_path + '/cmudict-en-us.dict'
# )
# text = ''
# for phrase in speech:
#     text += str(phrase)
# # write transcribed text to file
# with open('transcript.txt', 'w') as f:
#     f.write(text)
# # generate subtitle file
# subs = pysrt.SubRipFile()
# subs.append(pysrt.SubRipItem(index=1, text=text, start=0, end=len(text)))
# subs.save('subtitle.srt')

# pip install pytube >> run this in ipython console or in command propmp^

from pytube import YouTube 
import time
from sketchpy import library as lib

start_time = time.time()


obj= lib.vijay()
obj.draw()
link = 'https://www.youtube.com/watch?v=OfaBZvvL_7M'
yt = YouTube(link)

mp4_files = yt.streams.filter(file_extension="mp4")

mp4_369p_files = mp4_files.get_by_resolution("720p")

mp4_369p_files.download()



print("--- %s seconds ---" % (time.time() - start_time))
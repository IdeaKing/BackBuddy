from PIL import Image, ImageSequence
import vidaug.augmentors as va
import imageio
import cv2
from moviepy.editor import *

# load the video
path = 'D:/coding/BlairHacks_5/jason/'

files = os.listdir(path)

list = [va.HorizontalFlip(), va.VerticalFlip(), va.InvertColor(), va.Pepper()]
list2 = []

counter = 0
for fileName in files:
    if fileName.endswith('.mp4'):
        fileName = fileName[:-4]
        clip = VideoFileClip("D:/coding/BlairHacks_5/jason/" + fileName + ".mp4")


        # getting only 3 first seconds from video
        clip = clip.subclip(0, 15)

        # save the video clip as gif
        clip.write_gif("D:/coding/aug/vidaug/videos/" + fileName + ".gif")

        def gif_loader(path, modality="RGB"):
            frames = []
            with open(path, 'rb') as f:
                with Image.open(f) as video:
                    index = 1
                    for frame in ImageSequence.Iterator(video):
                        frames.append(frame.convert(modality))
                        index += 1
                return frames

        frames = gif_loader("D:/coding/aug/vidaug/videos/" + fileName + ".gif")
        sometimes = lambda aug: va.Sometimes(1, aug) # Used to apply augmentor with 100% probability

        for i in list:
            seq = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]
                sometimes(i) # horizontally flip the video with 100% probability
            ])

            video_aug = seq(frames)
            video_aug[0].save("D:/coding/BlairHacks_5/out.gif", save_all=True, append_images=video_aug[1:], duration=100, loop=0)

            vidcap = cv2.VideoCapture("D:/coding/BlairHacks_5/out.gif")
            success,image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite("D:/coding/BlairHacks_5/jasonimage/" + str(counter) + "-----" + fileName + "000000-%d.jpg" % count, image)     # save frame as JPEG file
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
            counter += 1

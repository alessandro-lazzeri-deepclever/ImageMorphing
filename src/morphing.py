import numpy as np
import cv2
from tqdm import tqdm
import argparse
import random
import math


def cross_dissolve_morphing(im_start, im_end, fps = 60, secs = 5):
    assert im_start.shape == im_end.shape
    im_start = np.array(im_start)
    im_end = np.array(im_end)
    frame_num = fps*secs
    for i in tqdm(range(frame_num)):
        frame = im_start*(1-i/(frame_num-1))+im_end*(i/(frame_num-1)) # (frame_num-1) last frame is im_end
        yield frame.astype(np.uint8)


def pixel_swap_morphing(im_start, im_end, fps = 60, secs = 5):
    assert im_start.shape == im_end.shape
    # shuffle all pixels coordinates
    pixels = [(i,j) for i in range(im_start.shape[0]) for j in range(im_start.shape[1])]
    random.shuffle(pixels)
    im_start = np.array(im_start)
    im_end = np.array(im_end)
    frame_num = fps*secs
    start = 0
    end = 0 #
    for i in tqdm(range(frame_num)):
        if end == 0:
            # first frame is the im_start
            frame = im_start
            end = start+int(math.ceil(len(pixels)/(frame_num-1)))
        else:
            end = start+int(math.ceil(len(pixels)/(frame_num-1)))
            if end > len(pixels):
                # final frame
                swap = pixels[start:]
            else:
                swap = pixels[start:end]
            for p in swap:
                frame[p] = im_end[p]
            start = end
        yield frame.astype(np.uint8)

if __name__ == '__main__':

    """
    im_start_path = Path(r'../dataset/1avROay.jpg')
    im_end_path = Path(r'../dataset/leGPvVl.jpg')
    video_path = Path(r'../result/video2.mp4')
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Source file name", default=None)
    parser.add_argument("-t", "--target", help="Target file name", default=None)
    parser.add_argument("-v", "--video", help="Output video file name", default='../video.mp4')
    parser.add_argument("-fps", "--fps", help="Frame per second", default=15, type=int)
    parser.add_argument("-l", "--time", help="Seconds of video", default=5, type=int)
    parser.add_argument("-m", "--mode", help="Type of morphing",
                    choices=['cross_dissolve', 'swap'], default='cross_dissolve')

    args = parser.parse_args()

    if not args.source:
        print("No source file provided!")
        exit()

    if not args.target:
        print("No target file provided!")
        exit()

    fps = args.fps
    im_start = cv2.imread(str(args.source))
    im_end = cv2.imread(str(args.target))
    assert im_start.shape == im_end.shape

    frameSize = (int(im_start.shape[1]), int(im_start.shape[0]))
    out = cv2.VideoWriter(str(args.video), 0, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)

    if args.mode == 'cross_dissolve':
        for frame in cross_dissolve_morphing(im_start, im_end, fps=fps, secs=args.time):
            out.write(frame)

    if args.mode == 'swap':
        for frame in pixel_swap_morphing(im_start, im_end, fps=fps, secs=args.time):
            out.write(frame)

    out.release()
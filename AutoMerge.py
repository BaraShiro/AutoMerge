import sys
import os
from typing import *
import numpy as np
import cv2 as cv
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from matplotlib import pyplot as plt
import time


# Gets frames from start to end from video at filepath and returns the in a list
def get_frames(start: int, number_of_frames_to_read: int, video: cv.VideoCapture, multichannel: bool = True) -> List[np.ndarray]:
    # Init variables
    out: List[np.ndarray] = []
    video.set(cv.CAP_PROP_POS_FRAMES, start)  # Jump to start frame

    # Read frames from the file, if reading the frame is successful append it to the list, else stop reading frames
    for x in range(number_of_frames_to_read):
        success, frame = video.read()
        if success:
            if multichannel:
                out.append(frame)
            else:
                out.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

        else:
            break

    return out


# Finds the most similar frames in the end of the lead vid and the beginning of the following vid
def find_matching_frames(lead_vid_path: str, following_vids_paths: List[str], seconds: int = 3,
                         multichannel: bool = True,  method: str = 'mse') -> List[Tuple[int, int, float]]:
    # TODO: catch file not found exceptions
    # Get lead video
    capture: cv.VideoCapture = cv.VideoCapture(lead_vid_path)
    number_of_frames: int = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps: int = int(capture.get(cv.CAP_PROP_FPS))
    number_of_frames_to_read: int = fps * seconds
    lead_vid_start: int = number_of_frames - number_of_frames_to_read
    lead_vid: List[np.ndarray] = get_frames(lead_vid_start, number_of_frames_to_read, capture, multichannel)
    capture.release()

    # Get following videos
    following_vids: List[List[np.ndarray]] = []
    for path in following_vids_paths:
        capture: cv.VideoCapture = cv.VideoCapture(path)
        fps: int = int(capture.get(cv.CAP_PROP_FPS))
        number_of_frames_to_read: int = fps * seconds
        following_vids.append(get_frames(0, number_of_frames_to_read, capture, multichannel))
        capture.release()

    # Calculate most similar frames
    out: List[Tuple[int, int, float]] = []
    for following_vid in following_vids:
        most_similar_frames: Tuple[int, int, float] = get_most_similar_frames(lead_vid, following_vid,
                                                                              lead_vid_start, multichannel, method)
        out.append(most_similar_frames)

    return out


def get_most_similar_frames(lead_vid: List[np.ndarray], following_vid: List[np.ndarray],
                               offset: int, multichannel: bool = True, method: str = 'mse') -> (int, int, float):
    if method == 'mse':
        min_diff = (0, 0, float('inf'))
        for i, lead_frame in enumerate(lead_vid):
            print("Processing frame", i, "of", len(lead_vid) - 1)
            for j, following_frame in enumerate(following_vid):
                diff: float = compare_mse(lead_frame, following_frame)
                if diff < min_diff[2]:
                    min_diff = (i + offset, j, diff)
        return min_diff

    elif method == 'nrmse':
        min_diff = (0, 0, float('inf'))
        for i, lead_frame in enumerate(lead_vid):
            print("Processing frame", i, "of", len(lead_vid) - 1)
            for j, following_frame in enumerate(following_vid):
                diff: float = compare_nrmse(lead_frame, following_frame)
                if diff < min_diff[2]:
                    min_diff = (i + offset, j, diff)
        return min_diff

    elif method == 'psnr':
        min_diff = (0, 0, float('-inf'))
        for i, lead_frame in enumerate(lead_vid):
            print("Processing frame", i, "of", len(lead_vid) - 1)
            for j, following_frame in enumerate(following_vid):
                diff: float = compare_psnr(lead_frame, following_frame)
                if diff > min_diff[2]:
                    min_diff = (i + offset, j, diff)
        return min_diff

    elif method == 'ssim':
        min_diff = (0, 0, float('-inf'))
        start = time.time()
        for i, lead_frame in enumerate(lead_vid):
            print("Processing frame", i, "of", len(lead_vid) - 1)
            for j, following_frame in enumerate(following_vid):
                diff = compare_ssim(lead_frame, following_frame, multichannel=multichannel)
                if diff > min_diff[2]:
                    min_diff = (i + offset, j, diff)
        end = time.time()
        print('Time elapsed:', end - start)
        return min_diff

    else:
        print("Invalid method, defaulting to MSE")
        return get_most_similar_frames(lead_vid, following_vid, offset, multichannel, 'mse')


def main():
    # capture: cv.VideoCapture = cv.VideoCapture('Test/red_blue_test.avi')
    # frames: List[np.ndarray] = get_frames(455, 460, capture)
    capture: cv.VideoCapture = cv.VideoCapture('Test/m.mkv')
    frames: List[np.ndarray] = get_frames(4000, 5000, capture)
    capture.release()
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    for f in frames:
        cv.imshow('frame', f)
        k = cv.waitKey(0) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()


# main()
print(find_matching_frames('Test/red_frame_test_1.avi', ['Test/red_frame_test_2.avi'], seconds=2, multichannel=False, method='mse'))
# print(find_matching_frames('Test/m.mkv', ['Test/m.mkv']))
# mao = cv.imread('Test/mao.jpg')
# maogray = cv.imread('Test/maogray.png')
# err = mean_square_error(mao, maogray)
# print(err)
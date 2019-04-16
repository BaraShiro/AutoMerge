import sys
import os
from typing import *
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Gets frames from start to end from video at filepath and returns the in a list
def get_frames(start: int, end: int, video: cv.VideoCapture) -> List[np.ndarray]:
    # Init variables
    out: List[np.ndarray] = []
    video.set(cv.CAP_PROP_POS_FRAMES, start)  # Jump to start frame

    # Read frames from the file, if reading the frame is successful append it to the list, else stop reading frames
    for x in range(end - start + 1):
        success, frame = video.read()
        if success:
            out.append(frame)
        else:
            break

    return out


# Finds the most similar frames in the end of the lead vid and the bigining of the following vid
def find_matching_frames(lead_vid_path: str, following_vids_paths: List[str],
                         seconds: int = 3, method: str = '') -> List[Tuple[int, int, float]]:
    # TODO: catch file not found exceptions
    capture: cv.VideoCapture = cv.VideoCapture(lead_vid_path)
    number_of_frames: float = capture.get(cv.CAP_PROP_FRAME_COUNT)
    fps: float = capture.get(cv.CAP_PROP_FPS)
    lead_vid_start: int = int(number_of_frames) - int((fps * seconds)) - 1
    lead_vid_end: int = int(number_of_frames) - 1
    print(number_of_frames, fps, lead_vid_start, lead_vid_end)
    lead_vid: List[np.ndarray] = get_frames(lead_vid_start, lead_vid_end, capture)
    capture.release()

    following_vids: List[List[np.ndarray]] = []
    for path in following_vids_paths:
        capture: cv.VideoCapture = cv.VideoCapture(path)
        fps: float = capture.get(cv.CAP_PROP_FPS)
        following_vid_end: int = int((fps * seconds)) - 1
        following_vids.append(get_frames(0, following_vid_end, capture))
        capture.release()

    out: List[Tuple[int, int, float]] = []
    for following_vid in following_vids:
        print("One vid")

        # Refactor this
        min_diff = (0, 0, float('inf'))  # TODO: confusing names, diff etc.?
        for i, lead_frame in enumerate(lead_vid):
            print("frame ", i)
            for j, following_frame in enumerate(following_vid):
                diff: float = get_image_difference(lead_frame, following_frame, method)
                if diff <= min_diff[2]:
                    min_diff = (i, j, diff)
        out.append(min_diff)

    return out


def get_image_difference(img_1: np.ndarray, img_2: np.ndarray, method: str = '') -> float:
    return mean_square_error(img_1, img_2)


def mean_square_error(lead_frame: np.ndarray, following_frame: np.ndarray) -> np.float64:
    error_sum: np.ndarray = np.sum((lead_frame.astype('float') - following_frame.astype('float')) ** 2, dtype=np.float64)
    image_size: int = lead_frame.shape[0] * lead_frame.shape[1]  # Assumes both images are the same size
    mean_error: np.float64 = error_sum / (image_size)
    return mean_error



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
print(find_matching_frames('Test/red_frame_test_1.avi', ['Test/red_frame_test_2.avi']))
# mao = cv.imread('Test/mao.jpg')
# maogray = cv.imread('Test/maogray.png')
# err = mean_square_error(mao, maogray)
# print(err)
import sys
import os
from typing import *
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Gets frames from start to end from video at filepath and returns the in a list
def get_frames(start: int, end:int, filepath: str) -> List[np.ndarray]:
    # Init variables
    out: List[np.ndarray] = []
    capture: cv.VideoCapture = cv.VideoCapture(filepath)
    # number_of_frames: float = capture.get(cv.CAP_PROP_FRAME_COUNT)
    capture.set(cv.CAP_PROP_POS_FRAMES, start)  # Jump to start frame

    # Read frames from the file, if reading the frame is successful append it to the list, else stop reading frames
    for x in range(end - start + 1):
        success, frame = capture.read()
        if success:
            out.append(frame)
        else:
            break

    capture.release()
    cv.destroyAllWindows()
    return out


frames: List[np.ndarray] = get_frames(455, 460, 'Test/red_blue_test.avi')
cv.namedWindow('frame', cv.WINDOW_NORMAL)
for f in frames:
    cv.imshow('frame', f)
    cv.waitKey()
cv.destroyAllWindows()
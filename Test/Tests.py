import sys
import os
from typing import *
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Generates a video with 499 blue frames and a single red frame at the end
def generate_red_blue_test_file() -> None:

    # Generate colored images
    red: np.ndarray = np.zeros((480, 640, 3), np.uint8)
    red[::] = (0,0,255)  # (B, G, R)
    blue: np.ndarray = np.zeros((480, 640, 3), np.uint8)
    blue[::] = (255,0,0)  # (B, G, R)

    # Set variables
    fourcc: int = cv.VideoWriter_fourcc(*'XVID') # Video format
    font: int = cv.FONT_HERSHEY_SIMPLEX
    number_of_frames: int = 500
    frames_per_second: int = 20.0

    # Init video writer
    out: cv.VideoWriter = cv.VideoWriter('red_blue_test.avi', fourcc, frames_per_second, (640, 480))
    # Make the blue frames
    for x in range(number_of_frames - 1):
        out_img: np.ndarray = np.copy(blue)
        # Put frame number on frame
        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)
    # Make the red frame
    cv.putText(red, str(number_of_frames - 1), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
    out.write(red)

    out.release()


# Generates two video files, a blue video with a red 485th frame, and a second green video with a red 15th frame
def generate_red_frame_test_files() -> None:

    # Generate colored images
    red: np.ndarray = np.zeros((480, 640, 3), np.uint8)
    red[::] = (0,0,255)  # (B, G, R)
    blue: np.ndarray = np.zeros((480, 640, 3), np.uint8)
    blue[::] = (255,0,0)  # (B, G, R)
    green: np.ndarray = np.zeros((480, 640, 3), np.uint8)
    green[::] = (0,255,0)  # (B, G, R)

    # Set variables
    fourcc: int = cv.VideoWriter_fourcc(*'XVID')
    font: int = cv.FONT_HERSHEY_SIMPLEX
    number_of_frames: int = 500
    frames_per_second: float = 20.0
    first_red_frame_number: int = 485
    second_red_frame_number: int = 15

    # Init video writer
    out: cv.VideoWriter = cv.VideoWriter('red_frame_test_1.avi', fourcc, frames_per_second, (640, 480))
    # Make the first video
    for x in range(number_of_frames):
        if x == first_red_frame_number:
            out_img: np.ndarray = np.copy(red)
        else:
            out_img: np.ndarray = np.copy(blue)

        # Put frame number on frame
        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)

    out.release()

    # Init video writer
    out = cv.VideoWriter('red_frame_test_2.avi', fourcc, frames_per_second, (640, 480))
    # Make the second video
    for x in range(number_of_frames):
        if x == second_red_frame_number:
            out_img: np.ndarray = np.copy(red)
        else:
            out_img: np.ndarray = np.copy(green)

        # Put frame number on frame
        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)

    out.release()


# Generates test data
def generate_test_data() -> None:
    # Check if files exist
    red_blue_exists: bool = os.path.isfile('red_blue_test.avi')
    red_frame_1_exists: bool = os.path.isfile('red_frame_test_1.avi')
    red_frame_2_exists: bool = os.path.isfile('red_frame_test_2.avi')

    # If files don't exist, create them
    if not red_blue_exists:
        generate_red_blue_test_file()

    if (not red_frame_1_exists) or (not red_frame_2_exists):
        generate_red_frame_test_files()


generate_test_data()

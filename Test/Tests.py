import sys
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def generate_red_blue_test_file():

    red = np.zeros((480, 640, 3), np.uint8)
    red[::] = (0,0,255)  # (B, G, R)
    blue = np.zeros((480, 640, 3), np.uint8)
    blue[::] = (255,0,0)  # (B, G, R)

    font = cv.FONT_HERSHEY_SIMPLEX
    number_of_frames = 500
    frames_per_second = 20.0

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('red_blue_test.avi', fourcc, frames_per_second, (640, 480))
    for x in range(number_of_frames - 1):
        out_img = np.copy(blue)
        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)
    cv.putText(red, str(number_of_frames - 1), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
    out.write(red)

    out.release()


def generate_red_frame_test_files():

    red = np.zeros((480, 640, 3), np.uint8)
    red[::] = (0,0,255)  # (B, G, R)
    blue = np.zeros((480, 640, 3), np.uint8)
    blue[::] = (255,0,0)  # (B, G, R)
    green = np.zeros((480, 640, 3), np.uint8)
    green[::] = (0,255,0)  # (B, G, R)

    font = cv.FONT_HERSHEY_SIMPLEX
    number_of_frames = 500
    frames_per_second = 20.0
    first_red_frame_number = 485
    second_red_frame_number = 15

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('red_frame_test_1.avi', fourcc, frames_per_second, (640, 480))
    for x in range(number_of_frames):
        if x == first_red_frame_number:
            out_img = np.copy(red)
        else:
            out_img = np.copy(blue)

        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)

    out.release()

    out = cv.VideoWriter('red_frame_test_2.avi', fourcc, frames_per_second, (640, 480))
    for x in range(number_of_frames):
        if x == second_red_frame_number:
            out_img = np.copy(red)
        else:
            out_img = np.copy(green)

        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)

    out.release()

def generate_test_data():
    red_blue_exists = os.path.isfile('red_blue_test.avi')
    red_frame_1_exists = os.path.isfile('red_frame_test_2.avi')
    red_frame_2_exists = os.path.isfile('red_frame_test_2.avi')
    if not red_blue_exists:
        generate_red_blue_test_file()

    if (not red_frame_1_exists) or (not red_frame_2_exists):
        generate_red_frame_test_files()

generate_test_data()
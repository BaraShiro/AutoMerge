import sys
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

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('red_blue_test.avi', fourcc, 20.0, (640, 480))
    for x in range(number_of_frames):
        out_img = np.copy(blue)
        cv.putText(out_img, str(x), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
        out.write(out_img)
    cv.putText(red, str(number_of_frames), (100, 400), font, 4, (255, 255, 255), 2, cv.LINE_AA)
    out.write(red)

    out.release()
    cv.destroyAllWindows()



generate_red_blue_test_file()
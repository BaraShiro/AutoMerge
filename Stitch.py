from typing import *
import numpy as np
import cv2 as cv
import os
from AutoMerge import get_frames, resize_image
from skimage.measure import compare_ssim



def stitch_videos(fst_vid_path: str, snd_vid_path: str, out_path: str,
                  fst_stitch_frame: int, snd_stitch_frame: int,
                  fst_seconds: int, snd_seconds: int) -> None:
    fst_capture: cv.VideoCapture = cv.VideoCapture(fst_vid_path)
    snd_capture: cv.VideoCapture = cv.VideoCapture(snd_vid_path)

    # Assuming same dimensions and fps on both videos
    fps: int = int(fst_capture.get(cv.CAP_PROP_FPS))
    width: float = fst_capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height: float = fst_capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    fst_number_of_frames_to_read: int = fps * fst_seconds
    snd_number_of_frames_to_read: int = fps * snd_seconds

    print("Getting", fst_number_of_frames_to_read, "frames from first video")
    fst_vid: List[np.ndarray] = get_frames(fst_stitch_frame - fst_number_of_frames_to_read + 1, fst_number_of_frames_to_read, fst_capture, True)
    print("Getting", snd_number_of_frames_to_read, "frames from second video")
    snd_vid: List[np.ndarray] = get_frames(snd_stitch_frame, snd_number_of_frames_to_read, snd_capture, True)

    fst_capture.release()
    snd_capture.release()

    second_file_name = snd_vid_path.split("/")[-1]
    out_name = second_file_name.split(".")[0]
    out_path += ("/" + out_name + " " + str(fst_stitch_frame) + " " + str(snd_stitch_frame) + "/")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # fourcc: int = cv.VideoWriter_fourcc(*'mp4v')  # Video format
    # out: cv.VideoWriter = cv.VideoWriter(out_path + out_name + '.mp4',
    #                                      fourcc, fps, (int(width), int(height)), True)

    # print("Stitching...")
    # for frame in fst_vid:
    #     out.write(frame)
    #
    # for frame in snd_vid:
    #     out.write(frame)
    #
    # out.release()

    print("Saving images...")

    fst_image = fst_vid[-1]
    snd_image = snd_vid[0]
    # fst_image = cv.cvtColor(fst_image, cv.COLOR_BGR2GRAY)
    # snd_image = cv.cvtColor(snd_image, cv.COLOR_BGR2GRAY)
    fst_image = resize_image(fst_image)
    snd_image = resize_image(snd_image)

    (score, diff_s_image) = compare_ssim(fst_image, snd_image, full=True, multichannel=True)

    diff_s_image = diff_s_image ** 2  # squared for visibility
    diff_s_image = (diff_s_image * 255).astype("uint8")
    # diff_s_image = cv.threshold(diff_s_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    diff_image = abs(fst_image.astype("int16") - snd_image.astype("int16")).astype("uint8")  # cast as int16 to avoid overflow
    diff_image_inv = np.invert(diff_image)  # Inverted for easy comparing with SSIM

    cv.imwrite(out_path + "first.jpg", fst_image)
    cv.imwrite(out_path + "second.jpg", snd_image)
    cv.imwrite(out_path + "diff_abs.jpg", diff_image_inv)
    cv.imwrite(out_path + "diff_ssim.jpg", diff_s_image)

    print("Done!")


out_dir = "out/"

# first_file = "Test/red_frame_test_1.avi"
# second_file = "Test/red_frame_test_2.avi"
# first_time = 485
# second_time = 15

first_file = "C:/360/Out/forest_2-1.mp4"
second_file = "C:/360/Out/forest_2-3.mp4"
# first_file = "G:/Camera/Video/Resolve/Path_2-1.mp4"
# second_file = "G:/Camera/Video/Resolve/Path_2-2.mp4"
first_time = 337
second_time = 87


stitch_videos(first_file, second_file, out_dir, first_time, second_time, 5, 5)


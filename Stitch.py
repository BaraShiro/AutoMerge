from typing import *
import numpy as np
import cv2 as cv
from AutoMerge import get_frames


def stitch_videos(fst_vid_path: str, snd_vid_path: str,
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

    fourcc: int = cv.VideoWriter_fourcc(*'XVID')  # Video format
    out: cv.VideoWriter = cv.VideoWriter('out/out.avi', fourcc, fps, (int(width), int(height)))

    print("stitching...")
    for frame in fst_vid:
        out.write(frame)

    for frame in snd_vid:
        out.write(frame)

    out.release()

    print("Done!")


stitch_videos("G:/Camera/Video/Garage_1_1.mp4", "G:/Camera/Video/Garage_1_2.mp4", 357, 136, 5, 5)

#stitch_videos("Test/red_frame_test_1.avi", "Test/red_frame_test_2.avi", 485, 15, 5, 5)
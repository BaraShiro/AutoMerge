from typing import *
import click
import numpy as np
import cv2 as cv
import os
from AutoMerge import get_frames, resize_image
from skimage.measure import compare_ssim


@click.command()
@click.option("--lead-file", "-l", "fst_vid_path", required=True,
              type=click.Path(exists=True, readable=True),
              help="Path to leading video file.")
@click.option("--follow-file", "-f", "snd_vid_path", required=True,
              type=click.Path(exists=True, readable=True),
              help="Path to following video file.")
@click.option("--out-dir", "-o", "out_path", required=True,
              type=click.Path(dir_okay=True),
              help="Path to out dir.")
@click.option("--lead-frame", "-x", "fst_stitch_frame", required=True, type=int,
              help="Frame from lead video to cut from.")
@click.option("--follow-frame", "-y", "snd_stitch_frame", required=True, type=int,
              help="Frame from following video to cut from.")
@click.option("--lead-len", "-z", "fst_seconds", type=int,
              help="Number of seconds from lead video.")
@click.option("--follow-len", "-w", "snd_seconds", type=int,
              help="Number of seconds from following video.")
def stitch_videos(fst_vid_path: str, snd_vid_path: str, out_path: str,
                  fst_stitch_frame: int, snd_stitch_frame: int,
                  fst_seconds: int = 5, snd_seconds: int = 5) -> None:
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

    second_file_dir, second_file_name = os.path.split(snd_vid_path)
    # second_file_name = snd_vid_path.split("/")[-1]
    out_name = str(second_file_name.split(".")[0])
    print(snd_vid_path, second_file_name, out_name)
    # out_path += ("/" + out_name + " " + str(fst_stitch_frame) + " " + str(snd_stitch_frame) + "/")
    out_dir_name = (out_name + " " + str(fst_stitch_frame) + " " + str(snd_stitch_frame))
    full_out_path = os.path.join(out_path, out_dir_name)
    if not os.path.isdir(full_out_path):
        os.mkdir(full_out_path)

    fourcc: int = cv.VideoWriter_fourcc(*'mp4v')  # Video format
    print(os.path.join(full_out_path, (out_name + '.mp4')))
    out: cv.VideoWriter = cv.VideoWriter(os.path.join(full_out_path, (out_name + '.mp4')),
                                         fourcc, fps, (int(width), int(height)), True)

    print("Stitching...")
    for frame in fst_vid:
        out.write(frame)

    for frame in snd_vid:
        out.write(frame)

    out.release()

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

    cv.imwrite(os.path.join(full_out_path, "first.jpg"), fst_image)
    cv.imwrite(os.path.join(full_out_path, "second.jpg"), snd_image)
    cv.imwrite(os.path.join(full_out_path, "diff_abs.jpg"), diff_image_inv)
    cv.imwrite(os.path.join(full_out_path, "diff_ssim.jpg"), diff_s_image)

    print("Done!")


if __name__ =="__main__":
    stitch_videos()

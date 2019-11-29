from typing import *
import click
import numpy as np
import cv2 as cv
import os
from AutoMerge import get_frames, resize_image
from skimage.measure import compare_ssim


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("fst_vid_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("snd_vid_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("fst_stitch_frame", type=click.IntRange(min=0, max=None, clamp=False))
@click.argument("snd_stitch_frame", type=click.IntRange(min=0, max=None, clamp=False))
@click.option("--lead-len", "-l", "fst_seconds", type=click.IntRange(min=1, max=None, clamp=False),
              help="Number of seconds from lead video, has to be greater than 0.")
@click.option("--follow-len", "-f", "snd_seconds", type=click.IntRange(min=1, max=None, clamp=False),
              help="Number of seconds from following video, has to be greater than 0.")
def stitch_videos(fst_vid_path: str, snd_vid_path: str,
                  fst_stitch_frame: int, snd_stitch_frame: int,
                  fst_seconds: int = 5, snd_seconds: int = 5) -> None:

    # Open in-video files
    fst_capture: cv.VideoCapture = cv.VideoCapture(fst_vid_path)
    snd_capture: cv.VideoCapture = cv.VideoCapture(snd_vid_path)

    # Create out dir
    second_file_dir, second_file_name = os.path.split(snd_vid_path)
    file_name, file_type = second_file_name.split(".")
    out_dir_name = (file_name + " " + str(fst_stitch_frame) + " " + str(snd_stitch_frame))
    full_out_path = os.path.join("out", out_dir_name)
    os.makedirs(full_out_path, exist_ok=True)

    # Get fps from in-video files
    fst_fps: int = int(fst_capture.get(cv.CAP_PROP_FPS))
    snd_fps: int = int(snd_capture.get(cv.CAP_PROP_FPS))
    # Calculate how many frames to read
    fst_number_of_frames_to_read: int = fst_fps * fst_seconds
    snd_number_of_frames_to_read: int = snd_fps * snd_seconds
    # Get width and height, assuming same dimensions on both in-video files
    width: float = fst_capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height: float = fst_capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    # Read frames from in-video files
    print("Getting", fst_number_of_frames_to_read, "frames from first video")
    fst_vid: List[np.ndarray] = get_frames(fst_stitch_frame - fst_number_of_frames_to_read + 1,
                                           fst_number_of_frames_to_read, fst_capture, True)
    print("Getting", snd_number_of_frames_to_read, "frames from second video")
    snd_vid: List[np.ndarray] = get_frames(snd_stitch_frame, snd_number_of_frames_to_read, snd_capture, True)

    # Close in-video files
    fst_capture.release()
    snd_capture.release()

    # Create out-video file, using same fps and dimensions as first in-video file
    fourcc: int = cv.VideoWriter_fourcc(*'mp4v')  # Video format
    out: cv.VideoWriter = cv.VideoWriter(os.path.join(full_out_path, (file_name + '.mp4')),
                                         fourcc, fst_fps, (int(width), int(height)), True)

    # Write frames to-out video file
    print("Stitching...")
    for frame in fst_vid:
        out.write(frame)

    for frame in snd_vid:
        out.write(frame)

    # Close out-video file
    out.release()

    print("Saving images...")
    # Get images
    fst_image = fst_vid[-1]
    snd_image = snd_vid[0]
    # Resize images
    fst_image = resize_image(fst_image)
    snd_image = resize_image(snd_image)

    # Calculate SSIM score and difference
    score, ssim_diff_image = compare_ssim(fst_image, snd_image, full=True, multichannel=True)
    # Square for visibility
    ssim_diff_image = ssim_diff_image ** 2
    # ssim_diff_image is float type, so convert back to uint8
    ssim_diff_image = (ssim_diff_image * 255).astype("uint8")

    # Calculate image absolute difference, cast as int16 to avoid overflow
    diff_image = abs(fst_image.astype("int16") - snd_image.astype("int16")).astype("uint8")
    # Invert for easy comparing with SSIM
    diff_image_inv = np.invert(diff_image)

    # Write images to disk
    cv.imwrite(os.path.join(full_out_path, "first.jpg"), fst_image)
    cv.imwrite(os.path.join(full_out_path, "second.jpg"), snd_image)
    cv.imwrite(os.path.join(full_out_path, "diff_abs.jpg"), diff_image_inv)
    cv.imwrite(os.path.join(full_out_path, "diff_ssim.jpg"), ssim_diff_image)

    print("Done!")


if __name__ =="__main__":
    stitch_videos()

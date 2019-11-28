import os
from typing import *
import numpy as np
import cv2 as cv
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
import time
import datetime
from joblib import Parallel, delayed


# Resizes an image to the new height, keeping the aspect ratio
def resize_image(image: np.ndarray, new_height: int = 480) -> np.ndarray:
    height: int = image.shape[0]
    width: int = image.shape[1]
    scale: float = new_height / height
    new_width: int = int(width * scale)
    new_image: np.ndarray = resize(image, (new_height, new_width), anti_aliasing=True)
    with warnings.catch_warnings():
        # Don't warn about loss of precision
        warnings.simplefilter("ignore")
        # resize() returns dtype float64, so convert back to uint8
        new_image = img_as_ubyte(new_image)
    return new_image


# Gets frames from start to end from video at filepath and returns the in a list
def get_frames(start: int, number_of_frames_to_read: int, video: cv.VideoCapture,
               multichannel: bool = True, downscale: bool = False, verbose: int = 0) -> List[np.ndarray]:
    # Init variables
    out: List[np.ndarray] = []
    # Jump to start frame
    video.set(cv.CAP_PROP_POS_FRAMES, start)

    # Read frames from the file,
    # if reading the frame is successful append it to the list, else stop reading frames
    for x in range(number_of_frames_to_read):
        success, frame = video.read()
        if success:

            if multichannel:
                out.append(frame)
            else:
                out.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

        else:
            if verbose >= 1:
                print("Not enough frames in video after frame number", start, "only", len(out), "frames read.")
            break

    if downscale:
        # Try to set number of jobs to the number of available CPUs.
        # If os.cpu_count() failed and returned None,
        # default to 4 jobs, as that's good enough
        number_of_jobs: int = os.cpu_count()
        if not number_of_jobs:
            number_of_jobs = 4

        if verbose >= 1:
            print("Resizing", number_of_frames_to_read, "frames, using", number_of_jobs, "threads...")

        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            resized_out = (parallel(delayed(resize_image)(frame, 480) for frame in out))

        return resized_out
    else:
        return out


# Finds the most similar frames in the end of the lead vid and the beginning of the following vid
# Verbose: 0 = nothing, 1 = stage of operation, 2 = threading and time, 3 = detailed processing
# Returns None if video file cannot be opened
def find_matching_frames(lead_vid_path: str, following_vids_paths: List[str], seconds: int,
                         multichannel: bool = True, downscale: bool = False,
                         method: str = 'mse', verbose: int = 0) -> Union[List[Tuple[int, int, float]], None]:

    if verbose >= 1:
        arg_message = "Processing " + str(seconds) + " seconds. Using " + method.upper()

        if multichannel:
            arg_message += ", color"
        else:
            arg_message += ", grayscale"

        if downscale:
            arg_message += ", and downscaling"
        else:
            arg_message += ", and original resolution"

        print(arg_message)

    start: float = time.time()

    # Get lead video
    capture: cv.VideoCapture = cv.VideoCapture(lead_vid_path)

    if not capture.isOpened():
        print("Error opening video file at", lead_vid_path)
        print("Make sure it exists, is a valid video file, and appropriate codecs are installed.")
        return None

    number_of_frames: int = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps: int = int(capture.get(cv.CAP_PROP_FPS))
    number_of_frames_to_read: int = fps * seconds
    lead_vid_start: int = number_of_frames - number_of_frames_to_read - 1

    if verbose >= 1:
        print("Getting", number_of_frames_to_read, "leading frames...")

    lead_vid: List[np.ndarray] = get_frames(lead_vid_start, number_of_frames_to_read, capture, multichannel, downscale, verbose)
    capture.release()

    # Get following videos
    following_vids: List[List[np.ndarray]] = []
    for path in following_vids_paths:
        capture: cv.VideoCapture = cv.VideoCapture(path)

        if not capture.isOpened():
            print("Error opening video file at", path)
            print("Make sure it exists, is a valid video file, and appropriate codecs are installed.")
            return None  # TODO: Maybe not the best way to handle this, consider making following_vids [] instead

        fps: int = int(capture.get(cv.CAP_PROP_FPS))
        number_of_frames_to_read: int = fps * seconds

        if verbose >= 1:
            print("Getting", number_of_frames_to_read, "following frames...")

        following_vids.append(get_frames(0, number_of_frames_to_read, capture, multichannel, downscale, verbose))
        capture.release()

    # Calculate most similar frames
    out: List[Tuple[int, int, float]] = []
    for following_vid in following_vids:
        most_similar_frames: Tuple[int, int, float] = get_most_similar_frames(lead_vid, following_vid,
                                                                              lead_vid_start, multichannel,
                                                                              method, verbose)
        out.append(most_similar_frames)

    end: float = time.time()
    if verbose >= 2:
        print('Time elapsed:', str(datetime.timedelta(seconds=(end - start))))

    return out


def run_mse(lead_frame: np.ndarray, following_frame: np.ndarray,
            lead_frame_number: int, following_frame_number: int) -> Tuple[int, int, float]:
    score: float = compare_mse(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


def run_nrmse(lead_frame: np.ndarray, following_frame: np.ndarray,
              lead_frame_number: int, following_frame_number: int) -> Tuple[int, int, float]:
    score: float = compare_nrmse(lead_frame, following_frame, norm_type="min-max")
    return lead_frame_number, following_frame_number, score


def run_psnr(lead_frame: np.ndarray, following_frame: np.ndarray,
             lead_frame_number: int, following_frame_number: int) -> Tuple[int, int, float]:
    score: float = compare_psnr(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


def run_ssim(lead_frame: np.ndarray, following_frame: np.ndarray,
             lead_frame_number: int, following_frame_number: int, multichannel: bool = True) -> Tuple[int, int, float]:
    score: float = compare_ssim(lead_frame, following_frame, multichannel=multichannel,
                                # Set arguments to match the implementation of Wang et. al.
                                gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
    return lead_frame_number, following_frame_number, score


def get_most_similar_frames(lead_vid: List[np.ndarray], following_vid: List[np.ndarray],
                            offset: int, multichannel: bool = True, method: str = 'mse',
                            verbose: int = 0) -> (int, int, float):

    number_of_jobs: int = os.cpu_count()    # Try to set number of jobs to the number of available CPUs.
    if not number_of_jobs:                  # If os.cpu_count() failed and returned None,
        number_of_jobs = 4                  # default to 4 jobs, as that's good enough

    if verbose >= 2:
        print("Processing", len(lead_vid), "frames, using", number_of_jobs, "threads...")

    if method == 'mse':
        min_diff: Tuple[int, int, float] = (0, 0, float('inf'))
        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            for i, lead_frame in enumerate(lead_vid):
                if verbose >= 3:
                    print("Processing frame", i + 1, "of", len(lead_vid))
                diff_list: List[Tuple[int, int, float]] = (parallel(delayed(run_mse)(lead_frame, following_frame, i + offset, j)
                                                                    for j, following_frame in enumerate(following_vid)))
                diff: Tuple[int, int, float] = min(diff_list, key=lambda diff_tuple: diff_tuple[2])
                if diff[2] < min_diff[2]:
                    min_diff = diff

        return min_diff

    elif method == 'nrmse':
        min_diff: Tuple[int, int, float] = (0, 0, float('inf'))
        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            for i, lead_frame in enumerate(lead_vid):
                if verbose >= 3:
                    print("Processing frame", i + 1, "of", len(lead_vid))
                diff_list: List[Tuple[int, int, float]] = (parallel(delayed(run_nrmse)(lead_frame, following_frame, i + offset, j)
                                                                    for j, following_frame in enumerate(following_vid)))
                diff: Tuple[int, int, float] = min(diff_list, key=lambda diff_tuple: diff_tuple[2])
                if diff[2] < min_diff[2]:
                    min_diff = diff

        return min_diff

    elif method == 'psnr':
        min_diff: Tuple[int, int, float] = (0, 0, float('-inf'))
        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            for i, lead_frame in enumerate(lead_vid):
                if verbose >= 3:
                    print("Processing frame", i + 1, "of", len(lead_vid))
                diff_list: List[Tuple[int, int, float]] = (parallel(delayed(run_psnr)(lead_frame, following_frame, i + offset, j)
                                                                    for j, following_frame in enumerate(following_vid)))
                diff: Tuple[int, int, float] = max(diff_list, key=lambda diff_tuple: diff_tuple[2])
                if diff[2] > min_diff[2]:
                    min_diff = diff

        return min_diff

    elif method == 'ssim':
        min_diff: Tuple[int, int, float] = (0, 0, float('-inf'))
        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            for i, lead_frame in enumerate(lead_vid):
                if verbose >= 3:
                    print("Processing frame", i + 1, "of", len(lead_vid))
                diff_list: List[Tuple[int, int, float]] = (parallel(delayed(run_ssim)(lead_frame, following_frame, i + offset, j, multichannel=multichannel)
                                                                    for j, following_frame in enumerate(following_vid)))
                diff: Tuple[int, int, float] = max(diff_list, key=lambda diff_tuple: diff_tuple[2])
                if diff[2] > min_diff[2]:
                    min_diff = diff

        return min_diff

    else:
        print("Invalid method, defaulting to MSE")
        return get_most_similar_frames(lead_vid, following_vid, offset, multichannel, 'mse')

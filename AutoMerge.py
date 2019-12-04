"""Searches and returns the best matching frames in two video files.

Given a leading video and a list of following videos, searches the end
of the leading and the beginning of the following  videos using one of four
image similarity metrics, MSE, NRMSE, PSNR, SSIM.

  Typical usage example:

  best_matching_frames = find_matching_frames("path/to/vid1.avi", ["path/to/vid2.avi", "path/to/vid3.avi"],
                                              seconds=3, multichannel=False, downscale=True,
                                              method="mse", verbose=3)
"""
import os
from typing import *
import click
from custom_params import PathList, Method
import numpy as np
import cv2 as cv
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
import time
import datetime
from joblib import Parallel, delayed


def resize_image(image: np.ndarray, new_height: int = 480) -> np.ndarray:
    """Resizes an image to the new height, keeping the aspect ratio.

    Resizes image, giving it a new height of new_height and a new width scaled
    accordingly to original aspect ratio, and converts it to uint8.
    Defaults to a new height of 480 pixels.

    Args:
        image: An images as an ndarray array.
        new_height: An optional variable that controls the height images is resized to.

    Returns:
        A resized version of image as an ndarray array with dtype uint8.

    Notes:
        Warning about precision loss is suppressed, but suppression might not always work.
    """

    height: int = image.shape[0]
    width: int = image.shape[1]
    scale: float = new_height / height
    new_width: int = int(width * scale)
    new_image: np.ndarray = resize(image, (new_height, new_width), anti_aliasing=True)
    with warnings.catch_warnings():
        # Don't warn about loss of precision when converting from uint8 to float64 and back to uint8
        warnings.simplefilter("ignore")
        # resize() returns dtype float64, so convert back to uint8
        new_image = img_as_ubyte(new_image)
    return new_image


def get_frames(start: int, number_of_frames_to_read: int, video: cv.VideoCapture,
               multichannel: bool = True, downscale: bool = False, verbose: int = 0) -> List[np.ndarray]:
    """Gets frames from video and returns them in a list.

    Gets number_of_frames_to_read number of frames starting from start
    from video and returns them in a list.
    The frames can be in colour or greyscale, and can optionally be downscaled to 480p.
    Can print detailed information on the process.

    Args:
        start: An int representing the frame number to start from.
        number_of_frames_to_read: An int representing the number of frames to read.
        video: An open OpenCV video capture to read frames from.
        multichannel: A bool for selecting to extract colour or greyscale frames.
        downscale: A bool for selecting to downscale extracted frames to 480p.
        verbose: An int controlling the printing of detailed information.
                 If verbose >= 1 and downscale == True,
                 the function prints how many frames are being resized and
                 how many threads are used to do the work.
                 If verbose >= 1 the function prints a notice
                 if the number of frames available after start is lower than
                 number_of_frames_to_read.

    Returns:
        A list of ndarrays with length equal to number_of_frames_to_read,
        containing frames from video, starting from frame number start.
        If the number of available frames after start is lower than
        number_of_frames_to_read the returned list will only contain so
        many frames as is available. If verbose is grater tha or equal to 1,
        a notice about this is printed.

    Notes:
        Uses at least 4 threads, but as most as many as the number of
        available logical processors, to resize images if downscale is enabled.
    """

    # Init return variable
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
                print("Only", len(out), "frames read, not enough frames in video after frame number", start)
            break

    if downscale:
        # Try to set number of jobs to the number of available CPUs.
        # If os.cpu_count() failed and returned None,
        # default to 4 jobs, as that's good enough
        number_of_jobs: int = os.cpu_count()
        if not number_of_jobs:
            number_of_jobs = 4

        if verbose >= 1:
            print("Resizing", len(out), "frames, using", number_of_jobs, "threads...")

        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            resized_out = (parallel(delayed(resize_image)(frame, 480) for frame in out))

        return resized_out
    else:
        return out


def find_matching_frames(lead_vid_path: str, following_vids_paths: List[str], seconds: int,
                         multichannel: bool = True, downscale: bool = False,
                         method: str = 'mse', verbose: int = 0) -> Union[List[Union[Tuple[int, int, float], None]], None]:
    """Finds the most similar frames in two videos.

    Searches the frames in the last seconds of the lead video
    and the first seconds of each of the following videos.
    The search can be made with four different methods (mse, nrmse, psnr, or ssim),
    in colour or greyscale, and with full resolution or downscaled resolution.

    Args:
        lead_vid_path: AString representing a path to the leading video file.
        following_vids_paths: A list of stings representing paths
        to the following video files.
        seconds: An int representing the number of seconds to search.
        multichannel: A bool for selecting to extract colour or greyscale frames.
        downscale: A bool for selecting to downscale extracted frames to 480p.
        method: A sting representing the image similarity method to use, valid values are:
                'mse': Mean squared error,
                'nrmse': Normalised root mean squared error,
                'psnr': peak signal-to-noise ratio,
                'ssim': Structural similarity measure.
                Defaults to 'mse'.
        verbose: An int controlling the printing of detailed information:
                 verbose <= 0 prints nothing,
                 verbose >= 1 prints stage of operation,
                 verbose >= 2 prints threading and time,
                 verbose >= 3 prints detailed processing.

    Returns:
        A list of int, int, float tuples or Nones, where each tuple or None is the result of
        a search on the leading video and one of the following videos,
        where the ints are the found frames and the float the similarity score,
        and a None means a following video could not be opened.
        Or None if the leading video could not be opened.
    """

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
            following_vids.append([])
        else:
            fps: int = int(capture.get(cv.CAP_PROP_FPS))
            number_of_frames_to_read: int = fps * seconds

            if verbose >= 1:
                print("Getting", number_of_frames_to_read, "following frames...")

            following_vids.append(get_frames(0, number_of_frames_to_read, capture, multichannel, downscale, verbose))
            capture.release()

    # Calculate most similar frames
    out: List[Tuple[int, int, float]] = []
    for following_vid in following_vids:
        if following_vid:
            most_similar_frames: Tuple[int, int, float] = get_most_similar_frames(lead_vid, following_vid,
                                                                                  lead_vid_start, multichannel,
                                                                                  method, verbose)
            out.append(most_similar_frames)
        else:
            out.append(None)

    end: float = time.time()
    if verbose >= 2:
        print('Time elapsed:', str(datetime.timedelta(seconds=(end - start))))

    return out


# Wrapper for skimage.measure.compare_mse().
def run_mse(lead_frame: np.ndarray, following_frame: np.ndarray,
            lead_frame_number: int, following_frame_number: int, _: bool = True) -> Tuple[int, int, float]:
    score: float = compare_mse(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


# Wrapper for skimage.measure.compare_nrmse().
def run_nrmse(lead_frame: np.ndarray, following_frame: np.ndarray,
              lead_frame_number: int, following_frame_number: int, _: bool = True) -> Tuple[int, int, float]:
    score: float = compare_nrmse(lead_frame, following_frame, norm_type="min-max")
    return lead_frame_number, following_frame_number, score


# Wrapper for skimage.measure.compare_psnr().
def run_psnr(lead_frame: np.ndarray, following_frame: np.ndarray,
             lead_frame_number: int, following_frame_number: int, _: bool = True) -> Tuple[int, int, float]:
    score: float = compare_psnr(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


# Wrapper for skimage.measure.compare_ssim().
def run_ssim(lead_frame: np.ndarray, following_frame: np.ndarray,
             lead_frame_number: int, following_frame_number: int, multichannel: bool = True) -> Tuple[int, int, float]:
    score: float = compare_ssim(lead_frame, following_frame, multichannel=multichannel,
                                # Set arguments to match the implementation of Wang et. al.
                                gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
    return lead_frame_number, following_frame_number, score


def compare_frames(lead_vid: List[np.ndarray], following_vid: List[np.ndarray], lower_is_better: bool,
                   compare_function: Callable[[np.ndarray, np.ndarray, int, int, bool], Tuple[int, int, float]],
                   offset: int, multichannel: bool = True, verbose: int = 0) -> (int, int, float):
    """Compares the frames in two lists of frames, and returns the most similar.

    Searches lead_vid and following_vid for the most similar frames
    using the compare function passed in compare_function and
    initial_value as the starting best score.

    Args:
        lead_vid: A list of ndarrays representing frames from the leading video
        following_vid: A list of ndarrays representing frames from the following video
        offset: An int representing the offset of the leading frames in the leading video,
                used to return the correct frame number for the leading video.
        lower_is_better: A bool indicating if lower scores are better or worse.
        compare_function: A function that compares frames and returns their similarity score and their numbers.
        multichannel: A bool specifying if the frames are in colour or greyscale,
                      passed to the SSIM function for it to function correctly.
        verbose: An int controlling the printing of detailed information:
                 verbose <= 1 prints nothing,
                 verbose >= 2 prints how many frames are being processed and how many thread used,
                 verbose >= 3 prints which out of how many frames are being processed.

    Returns:
        An tuple with two ints representing the frame numbers of the two most similar frames,
        and a  float representing the similarity score.
    """

    # Try to set number of jobs to the number of available CPUs.
    # If os.cpu_count() failed and returned None,
    # default to 4 jobs, as that's good enough.
    number_of_jobs: int = os.cpu_count()
    if not number_of_jobs:
        number_of_jobs = 4

    # If lower scores are better define better score as less than,
    # best score in list as min, and initial value as infinity
    if lower_is_better:
        def new_score_is_better(new: float, old: float) -> bool:
            return new < old

        get_best_score_in_list: Callable[[List[Tuple[int, int, float]]], Tuple[int, int, float]] = min
        initial_value: float = float('inf')

    # If higher scores are better define better score as greater than,
    # best score in list as max, and initial value to negative infinity.
    else:
        def new_score_is_better(new: float, old: float) -> bool:
            return new > old

        get_best_score_in_list: Callable[[List[Tuple[int, int, float]]], Tuple[int, int, float]] = max
        initial_value: float = float('-inf')

    if verbose >= 2:
        print("Processing", len(lead_vid), "frames, using", number_of_jobs, "threads...")

    best_score: Tuple[int, int, float] = (0, 0, initial_value)
    with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
        for i, lead_frame in enumerate(lead_vid):
            if verbose >= 3:
                print("Processing frame", i + 1, "of", len(lead_vid))
            score_list: List[Tuple[int, int, float]] = (parallel(delayed(compare_function)
                                                                 (lead_frame, following_frame, i + offset, j,
                                                                  multichannel)
                                                                 for j, following_frame in enumerate(following_vid)))
            new_score: Tuple[int, int, float] = get_best_score_in_list(score_list, key=lambda score: score[2])
            if new_score_is_better(new_score[2], best_score[2]):
                best_score = new_score

    return best_score


def get_most_similar_frames(lead_vid: List[np.ndarray], following_vid: List[np.ndarray],
                            offset: int, multichannel: bool = True, method: str = 'mse',
                            verbose: int = 0) -> (int, int, float):
    """Gets the most similar frames from two lists of frames.

    Searches lead_vid and following_vid for the most similar frames
    using the method specified in method.

    Args:
        lead_vid: A list of ndarrays representing frames from the leading video
        following_vid: A list of ndarrays representing frames from the following video
        offset: An int representing the offset of the leading frames in the leading video,
                used to return the correct frame number for the leading video.
        multichannel: A bool specifying if the frames are in colour or greyscale,
                      passed to the SSIM function for it to function correctly.
        method: A sting representing the image similarity method to use, valid values are:
                'mse': Mean squared error,
                'nrmse': Normalised root mean squared error,
                'psnr': peak signal-to-noise ratio,
                'ssim': Structural similarity measure.
                Defaults to 'mse'.
        verbose: An int controlling the printing of detailed information:
                 verbose <= 1 prints nothing,
                 verbose >= 2 prints how many frames are being processed and how many thread used,
                 verbose >= 3 prints which out of how many frames are being processed.

    Returns:
        An tuple with two ints representing the frame numbers of the two most similar frames,
        and a  float representing the similarity score.
    """

    if method == 'mse':
        return compare_frames(lead_vid, following_vid, True, run_mse, offset, multichannel, verbose)

    elif method == 'nrmse':
        return compare_frames(lead_vid, following_vid, True, run_nrmse, offset, multichannel, verbose)

    elif method == 'psnr':
        return compare_frames(lead_vid, following_vid, False, run_psnr, offset, multichannel, verbose)

    elif method == 'ssim':
        return compare_frames(lead_vid, following_vid, False, run_ssim, offset, multichannel, verbose)

    else:
        print("Invalid method, defaulting to MSE")
        return get_most_similar_frames(lead_vid, following_vid, offset, multichannel, 'mse')


@click.command(context_settings={"ignore_unknown_options": True}, options_metavar='<options>')
@click.argument("lead_vid_path", type=click.Path(exists=True, dir_okay=False, readable=True), metavar='<leading video>')
@click.argument("following_vids_paths", type=PathList(), metavar='<following videos>')
@click.argument("seconds", type=click.IntRange(min=1, max=None, clamp=False), metavar='<seconds>')
@click.argument("method", type=Method(), metavar='<method>')
@click.option("--verbose", type=click.IntRange(min=0, max=3, clamp=False), default=0,
              help='0 = nothing, 1 = stage of operation, 2 = threading and time, 3 = detailed processing')
@click.option('--colour/--greyscale', default=False, help='colour on / off (default off)')
@click.option('--downscale/--no-downscale', default=True, help='downscale on / off (default on)')
def driver(lead_vid_path: str, following_vids_paths: List[str], seconds: int,
           colour, downscale, method: str, verbose: int) -> None:
    """Finds the best matching frames in the <seconds> last seconds of <leading video>
    and the <seconds> first seconds of <following videos>, using <methods> as similarity measure.

    <leading video> is the path to the leading video.

    <following videos> is the path or a comma separated list of paths to one or several following videos.

    <seconds> is the number of seconds to search.

    <method> is the similarity measure to use. Valid options are: mse, nrmse, psnr, ssim.
    """
    print(find_matching_frames(lead_vid_path, following_vids_paths, seconds, colour, downscale, method, verbose))


if __name__ == "__main__":
    driver()

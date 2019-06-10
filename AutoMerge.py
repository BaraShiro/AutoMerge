import os
from typing import *
import numpy as np
import cv2 as cv
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
import time
from joblib import Parallel, delayed


# Resizes an image to the new with, keeping the aspect ratio
def resize_image(image: np.ndarray, new_width: int = 640) -> np.ndarray:
    height: int = image.shape[0]
    width: int = image.shape[1]
    scale: float = new_width / width
    new_height: int = int(height * scale)
    new_image: np.ndarray = resize(image, (new_height, new_width), anti_aliasing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Don't warn about loss of precision
        new_image = img_as_ubyte(new_image)  # resize() returns dtype float64, so covert back to uint8
    return new_image



# Gets frames from start to end from video at filepath and returns the in a list
def get_frames(start: int, number_of_frames_to_read: int, video: cv.VideoCapture,
               multichannel: bool = True, downscale: bool = False) -> List[np.ndarray]:
    # Init variables
    out: List[np.ndarray] = []
    video.set(cv.CAP_PROP_POS_FRAMES, start)  # Jump to start frame

    if downscale:
        print("Resizing", number_of_frames_to_read, "frames")

    # Read frames from the file, if reading the frame is successful append it to the list, else stop reading frames
    for x in range(number_of_frames_to_read):
        success, frame = video.read()
        if success:

            if downscale:
                frame = resize_image(frame)

            if multichannel:
                out.append(frame)
            else:
                out.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

        else:
            break

    return out


# Finds the most similar frames in the end of the lead vid and the beginning of the following vid
def find_matching_frames(lead_vid_path: str, following_vids_paths: List[str], seconds: int,
                         multichannel: bool = True, downscale: bool = False,
                         method: str = 'mse') -> List[Tuple[int, int, float]]:
    # TODO: catch file not found exceptions
    # TODO: check for seconds longer that video length
    start = time.time()

    # Get lead video
    capture: cv.VideoCapture = cv.VideoCapture(lead_vid_path)
    number_of_frames: int = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps: int = int(capture.get(cv.CAP_PROP_FPS))
    number_of_frames_to_read: int = fps * seconds
    lead_vid_start: int = number_of_frames - number_of_frames_to_read
    print("Getting", number_of_frames_to_read, "leading frames")
    lead_vid: List[np.ndarray] = get_frames(lead_vid_start, number_of_frames_to_read, capture, multichannel, downscale)
    capture.release()

    # Get following videos
    following_vids: List[List[np.ndarray]] = []
    for path in following_vids_paths:
        capture: cv.VideoCapture = cv.VideoCapture(path)
        fps: int = int(capture.get(cv.CAP_PROP_FPS))
        number_of_frames_to_read: int = fps * seconds
        print("Getting", number_of_frames_to_read, "following frames")
        following_vids.append(get_frames(0, number_of_frames_to_read, capture, multichannel, downscale))
        capture.release()

    # Calculate most similar frames
    out: List[Tuple[int, int, float]] = []
    for following_vid in following_vids:
        most_similar_frames: Tuple[int, int, float] = get_most_similar_frames(lead_vid, following_vid,
                                                                              lead_vid_start, multichannel, method)
        out.append(most_similar_frames)

    end = time.time()
    print('Time elapsed:', end - start)

    return out


def run_mse(lead_frame: np.ndarray, following_frame: np.ndarray,
            lead_frame_number: int, following_frame_number: int) -> Tuple[int, int, float]:
    score: float = compare_mse(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


def run_nrmse(lead_frame: np.ndarray, following_frame: np.ndarray,
              lead_frame_number: int, following_frame_number: int) -> Tuple[int, int, float]:
    score: float = compare_nrmse(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


def run_psnr(lead_frame: np.ndarray, following_frame: np.ndarray,
             lead_frame_number: int, following_frame_number: int) -> Tuple[int, int, float]:
    score: float = compare_psnr(lead_frame, following_frame)
    return lead_frame_number, following_frame_number, score


def run_ssim(lead_frame: np.ndarray, following_frame: np.ndarray,
             lead_frame_number: int, following_frame_number: int, multichannel: bool = True) -> Tuple[int, int, float]:
    score: float = compare_ssim(lead_frame, following_frame, multichannel=multichannel)
    return lead_frame_number, following_frame_number, score


def get_most_similar_frames(lead_vid: List[np.ndarray], following_vid: List[np.ndarray],
                               offset: int, multichannel: bool = True, method: str = 'mse') -> (int, int, float):

    number_of_jobs: int = os.cpu_count()  # Try to set number of jobs to the number of available CPUs
    if not number_of_jobs:  # If os.cpu_count() failed and returned None
        number_of_jobs = 4  # Default to 4 jobs, as that's good enough

    if method == 'mse':
        min_diff: Tuple[int, int, float] = (0, 0, float('inf'))
        with Parallel(n_jobs=number_of_jobs, prefer="threads") as parallel:
            for i, lead_frame in enumerate(lead_vid):
                print("Processing frame", i, "of", len(lead_vid) - 1)
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
                print("Processing frame", i, "of", len(lead_vid) - 1)
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
                print("Processing frame", i, "of", len(lead_vid) - 1)
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
                print("Processing frame", i, "of", len(lead_vid) - 1)
                diff_list: List[Tuple[int, int, float]] = (parallel(delayed(run_ssim)(lead_frame, following_frame, i + offset, j, multichannel=multichannel)
                                                                    for j, following_frame in enumerate(following_vid)))
                diff: Tuple[int, int, float] = max(diff_list, key=lambda diff_tuple: diff_tuple[2])
                if diff[2] > min_diff[2]:
                    min_diff = diff

        return min_diff

    else:
        print("Invalid method, defaulting to MSE")
        return get_most_similar_frames(lead_vid, following_vid, offset, multichannel, 'mse')


def main():
    # capture: cv.VideoCapture = cv.VideoCapture('Test/red_blue_test.avi')
    # frames: List[np.ndarray] = get_frames(455, 460, capture)
    capture: cv.VideoCapture = cv.VideoCapture('Test/m.mkv')
    frames: List[np.ndarray] = get_frames(4000, 5000, capture)
    capture.release()
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    for f in frames:
        cv.imshow('frame', f)
        k = cv.waitKey(0) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()


# main()
# print(find_matching_frames('Test/red_frame_test_1.avi', ['Test/red_frame_test_2.avi'], seconds=2, multichannel=False, method='mse'))
# print(find_matching_frames('Test/m.mkv', ['Test/m.mkv']))
# mao = cv.imread('Test/mao.jpg')
# maogray = cv.imread('Test/maogray.png')
# err = mean_square_error(mao, maogray)
# print(err)
#print(find_matching_frames('Test/red_frame_test_1.avi', ['Test/red_frame_test_2.avi'], seconds=1, multichannel=False, method='psnr'))

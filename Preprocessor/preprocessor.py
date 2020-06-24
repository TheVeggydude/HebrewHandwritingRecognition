import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

from scipy.signal import find_peaks
from Preprocessor.utils import count_transitions
from Preprocessor.path_finding import find_path


# Constants
ROWS = 0
COLUMNS = 1


# def extract_path(x1, x2, y1, y2):
#     upper_bound = min(x1)
#     lower_bound = max(x2)
#     extracted_image = np.ones((lower_bound - upper_bound, line_image.shape[1]), dtype=int) * 255
#     extracted_image = line_image[upper_bound:lower_bound, :]
#     for x, y in zip(x1, y1):
#         extracted_image[0: x - upper_bound, y] = 255
#     for x, y in zip(x2, y2):
#         extracted_image[x - upper_bound:, y] = 255
#     return extracted_image


def binarize_image(image):
    """
    Binarizes a given cv2 image using Otsu's method after Gaussian blurring.
    :param image: cv2 (opencv-python) image.
    :return: binarized cv2 image.
    """

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def find_line_starts(projection):
    """
    Finds the start of line segmentation by taking the center minimal value between peaks.
    :param projection: list of projection values. Each element corresponds to a row in the original data.
    :return: list of tuples, consisting of row indices and a set of peaks the row is in between.
    """
    peaks, properties = find_peaks(projection, prominence=1, distance=100)

    start_data = []
    for index, peak in enumerate(peaks[:-1]):
        subset = projection[peaks[index]:peaks[index+1]]
        local_minima = np.where(subset == np.amin(subset))[0]
        start_data.append((peak + local_minima[int(len(local_minima)/2)], [peak, peaks[index+1]]))

    return start_data


if __name__ == '__main__':

    for i in range(0, 3):
        # Load and binarize image
        print(f"Working on test image {i}")
        img = cv2.imread(f"../data/test{i}.jpg", 0)
        img_binarized = binarize_image(img)

        # Compute projection & find line starts.
        img_arr = np.array(img_binarized)
        print(f"Image shape: {img_arr.shape}")
        projection = np.apply_along_axis(count_transitions, COLUMNS, img_arr)
        line_start_data = find_line_starts(projection)

        for index, start in enumerate(line_start_data):
            t = time.time()
            # do stuff

            print(f"Finding path for line at row {start} ({index+1}/{len(line_start_data)}).")
            path = find_path(start[0], start[1], img_arr)
            print(f"Path found in {time.time() - t} seconds!")

            # TODO verify path integrity
            plt.plot(path[:, COLUMNS], path[:, ROWS])
        plt.imshow(img_arr, 'gray')
        plt.show()

        # plt.imshow(extracted_image, 'gray', vmin=0)
        # plt.show()
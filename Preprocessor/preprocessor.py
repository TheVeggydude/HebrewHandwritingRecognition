import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

from scipy.signal import find_peaks
from Preprocessor.utils import count_transitions
from Preprocessor.path_finding import find_path, MOVES


# Constants
ROWS = 0
COLUMNS = 1


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


def crop_segment(upper, lower, img_data, index=0):
    """

    :param upper:
    :param lower:
    :param img_data:
    :return:
    """

    # Find boundary values
    upper_bound = min(upper[:, 0]) if upper is not None else 0  # min because the highest row has the lowest value
    lower_bound = max(lower[:, 0]) if lower is not None else img_data.shape[ROWS]-1

    # Crop horizontally
    cropped = img_data[upper_bound: lower_bound, :]

    if upper is not None:
        for bound in upper:
            cropped[:bound[ROWS]-upper_bound, bound[COLUMNS]] = 255  # set everything above boundary to white

    if lower is not None:
        for bound in lower:
            cropped[bound[ROWS]-upper_bound:, bound[COLUMNS]] = 255  # set everything below boundary to white

    plt.imshow(cropped, 'gray')
    plt.title(f"Segment {index}")
    plt.show()

    return cropped


def segment(img):
    # Load and binarize image
    img_binarized = binarize_image(img)

    # Compute projection & find line starts.
    img_data = np.array(img_binarized)
    print(f"Image shape: {img_data.shape}")
    projection = np.apply_along_axis(count_transitions, COLUMNS, img_data)
    line_start_data = find_line_starts(projection)

    # Find all the segments
    segmentation_lines = [None for elem in line_start_data]  # to store the segmentation lines
    for index, start in enumerate(line_start_data):
        t = time.time()
        print(f"Finding path for line at row {start} ({index + 1}/{len(line_start_data)}).")

        segmentation_lines[index] = find_path(start[0], start[1], img_data)
        print(f"Path found in {time.time() - t} seconds!")

        plt.plot(segmentation_lines[index][:, COLUMNS], segmentation_lines[index][:, ROWS])

    # Plot the segmentation
    plt.imshow(img_data, 'gray')
    plt.title(f"Image {i} segmentation")
    plt.show()

    # Actually split the image into segments
    upper = None
    for index, lower in enumerate(segmentation_lines):
        crop_segment(upper, lower, img_data, index)
        upper = lower

        if index == len(segmentation_lines) - 1:
            print("final one")
            crop_segment(upper, None, img_data, index+1)


if __name__ == '__main__':

    for i in range(2, 3):
        print(f"Working on test image {i}")
        img = cv2.imread(f"../data/test{i}.jpg", 0)
        segment(img)

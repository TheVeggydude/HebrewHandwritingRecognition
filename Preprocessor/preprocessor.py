import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

from scipy.signal import find_peaks
from Preprocessor.utils import count_transitions, count_ink
from Preprocessor.path_finding import find_path, extract_sub_image


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


def find_line_starts(projection, distance):
    """
    Finds the start of line segmentation by taking the center minimal value between peaks.
    :param projection: list of projection values. Each element corresponds to a row in the original data.
    :return: list of tuples, consisting of row indices and a set of peaks the row is in between.
    """
    peaks, properties = find_peaks(projection, prominence=1, distance=distance)

    start_data = []
    for index, peak in enumerate(peaks[:-1]):
        subset = projection[peaks[index]:peaks[index+1]]
        local_minima = np.where(subset == np.amin(subset))[0]
        start_data.append((peak + local_minima[int(len(local_minima)/2)], [peak, peaks[index+1]]))

    return start_data


def cut_out_segment(upper, lower, img_data):
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

    return cropped


def segment_sentences(image, debug=False):
    """

    :param image:
    :param debug:
    :return:
    """
    # Load and binarize image
    image = binarize_image(image)

    # Compute projection & find line starts.
    image = np.array(image)
    print(f"Image shape: {image.shape}")
    projection = np.apply_along_axis(count_transitions, COLUMNS, image)
    line_start_data = find_line_starts(projection, 100)

    # Find all the segments
    segmentation_lines = [None for elem in line_start_data]  # to store the segmentation lines
    for index, start in enumerate(line_start_data):
        t = time.time()
        print(f"Finding path for line at row {start} ({index + 1}/{len(line_start_data)}).")

        segmentation_lines[index] = find_path(start[0], start[1], image, debug)
        print(f"Path found in {time.time() - t} seconds!")

        if debug:
            plt.plot(segmentation_lines[index][:, COLUMNS], segmentation_lines[index][:, ROWS])

    # Plot the segmentation
    if debug:
        plt.imshow(image, 'gray')
        plt.title(f"Image segmentation")
        plt.show()

    # Actually split the image into segments
    segments = []
    upper = None
    for index, lower in enumerate(segmentation_lines):
        segments.append(cut_out_segment(upper, lower, image))
        upper = lower

        if index == len(segmentation_lines) - 1:
            segments.append(cut_out_segment(upper, None, image))

    return segments


def valid_path(path):
    """
    Validates a path.
    :param path:
    :return:
    """
    return path is not None


def segment_characters(image, debug=False):
    """

    :param image:
    :param debug:
    :return:
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        ink_projection = np.apply_along_axis(count_ink, COLUMNS, image)
        transition_projection = np.apply_along_axis(count_transitions, COLUMNS, image)
        ratio_projection = np.nan_to_num(np.divide(ink_projection, transition_projection))

    line_starts = find_line_starts(ratio_projection, 15)

    # Find all the segments
    segmentation_lines = []  # to store the segmentation lines
    for index, start in enumerate(line_starts):
        t = time.time()
        print(f"Finding path for line at row {start} ({index + 1}/{len(line_starts)}).")

        path = find_path(start[0], start[1], image, debug)
        if valid_path(path):
            segmentation_lines.append(path)
            print(f"Path found in {time.time() - t} seconds!")
        else:
            print("Invalid path found!")

    if debug:
        peaks, properties = find_peaks(ratio_projection, prominence=1, distance=15)
        line_start_rows = [start[0] for start in line_starts]

        # # Plot the projection
        # plt.plot(ratio_projection, 'red')
        # plt.plot(peaks, ratio_projection[peaks], "x")
        # plt.plot(line_start_rows, ratio_projection[line_start_rows], "o")
        #
        # plt.title("Projections")
        # plt.legend(['Ink/Transitions', 'peaks'])
        # plt.show()

        # Plot the segmentation lines
        for line in segmentation_lines:
            plt.plot(line[:, COLUMNS], line[:, ROWS])

        plt.imshow(image, 'gray')
        plt.title(f"Image segmentation")
        plt.show()


if __name__ == '__main__':

    for i in range(0, 3):
        print(f"Working on test image {i}")
        img = cv2.imread(f"../data/test{i}.jpg", 0)

        # Get image sentences
        sentences = segment_sentences(img)

        # Get the characters per sentence
        for sentence in sentences:

            # Crop sentence image to relevant area
            _, _, _, sentence = extract_sub_image(sentence)
            _, _, _, sentence = extract_sub_image(np.transpose(sentence))

            segment_characters(sentence)

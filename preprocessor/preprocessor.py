import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

from scipy.signal import find_peaks
from pyprind import ProgBar
from utils import count_transitions, count_ink, extract_sub_image, COLUMNS, ROWS, MOVES
from path_finding import find_path


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
    projection = np.apply_along_axis(count_transitions, COLUMNS, image)
    line_start_data = find_line_starts(projection, 100)

    # Storage
    segmentation_lines = []  # to store the segmentation lines
    sentences = []  # for storing the actual segments
    characters = []

    # Find and store the segments
    bar = ProgBar(len(line_start_data)+1, track_time=False, stream=sys.stdout, title='Segmenting sentences')
    bar.update()
    upper_line = None
    for index, start in enumerate(line_start_data):

        # Find and cut out segment
        line = find_path(start[0], start[1], image, debug)
        segmentation_lines.append(line)

        # Store the segment
        sentence = cut_out_segment(upper_line, line, image)
        sentences.append(sentence)
        upper_line = line

        # One extra segment after the last line.
        if index == len(line_start_data) - 1:
            sentences.append(cut_out_segment(upper_line, None, image))

        bar.update()

    if debug:

        # Plot the segmentation lines
        for line in segmentation_lines:
            plt.plot(line[:, COLUMNS], line[:, ROWS])

        plt.imshow(image, 'gray')
        plt.title(f"Image segmentation")
        plt.show()

    return sentences


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

        path = find_path(start[0], start[1], image, debug)
        if valid_path(path):
            segmentation_lines.append(path)

    # Actually split the image into segments
    segments = []
    upper = None
    for index, lower in enumerate(segmentation_lines):
        segments.append(np.transpose(cut_out_segment(upper, lower, image)))
        upper = lower

        if index == len(segmentation_lines) - 1:
            segments.append(np.transpose(cut_out_segment(upper, None, image)))

    if debug:

        # Plot the segmentation lines
        for line in segmentation_lines:
            plt.plot(line[:, COLUMNS], line[:, ROWS])

        plt.imshow(image, 'gray')
        plt.title(f"Image segmentation")
        plt.show()

    return segments


def get_characters_from_image(filename, debug=False):
    print(f"Working on {filename}")
    image = cv2.imread(filename, 0)

    # Get image sentences
    sentences = segment_sentences(image, debug)

    # Get the characters per sentence
    characters = None
    for sentence in sentences:

        # Crop sentence image to relevant area
        _, _, _, sentence = extract_sub_image(sentence)
        _, _, _, sentence = extract_sub_image(np.transpose(sentence))

        # Find the characters
        new_chars = segment_characters(sentence, debug)
        if not characters:
            characters = new_chars
        else:
            characters = characters + new_chars

    if debug:
        for index, character in enumerate(characters):
            cv2.imwrite(f"results/characters/test{i}_character_{index}.jpg", character)


if __name__ == '__main__':

    for i in range(0, 1):
        file = f"../data/test{i}.jpg"
        get_characters_from_image(file, debug=True)

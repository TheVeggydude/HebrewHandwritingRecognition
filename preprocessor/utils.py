import numpy as np
from collections import namedtuple


# Constants
ROWS = 0
COLUMNS = 1
WINDOW_BUFFER = 10
MOVES = [[0, 1], [1, 1], [-1, 1], [1, 0], [-1, 0]]

# Named tuple definitions
Point = namedtuple("Point", ["x", "y"])


def count_transitions(row):
    """
    Count the number of transitions between values in an image row.
    :param row: row in binarized image.
    :return: number of transitions in row.
    """

    return len(np.argwhere(np.diff(row)))


def count_ink(row):
    """
    Count the number of inked pixels in an image row.
    :param row: row in binarized image.
    :return: number of inked pixels in row.
    """

    return len(np.where(row == 0)[0])


def extract_sub_image(image, limits=(0, -1)):
    """
    Creates view of the image NumPy array cropped of the useless whitespace on the left and right sides of the image.
    :param image: 2D NumPy array describing the image data.
    :param limits: List containing two row limits to crop to.
    :return: Cropped view of the original 2D NumPy array.
    """
    max_width = np.shape(image)[COLUMNS] - 1
    relevant_columns = [0, max_width]

    cropped_img = image[limits[0]: limits[1], :]

    # Find first column that shows a pixel.
    for column in range(cropped_img.shape[COLUMNS]):
        transitions = count_transitions(cropped_img[:, column])
        if transitions > 0:
            relevant_columns[0] = column
            break

    # Find last column that shows a pixel.
    for column in range(cropped_img.shape[COLUMNS] - 1, 0, -1):
        transitions = count_transitions(cropped_img[:, column])
        if transitions > 0:
            relevant_columns[1] = column
            break

    # Compute margins
    left_margin = relevant_columns[0]
    left_margin = left_margin - WINDOW_BUFFER if left_margin - WINDOW_BUFFER > 0 else 0

    right_margin = relevant_columns[1]
    right_margin = right_margin + WINDOW_BUFFER if right_margin + WINDOW_BUFFER < max_width else max_width

    return left_margin, right_margin, limits[0], image[limits[0]:limits[1], left_margin: right_margin]

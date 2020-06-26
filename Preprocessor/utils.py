import numpy as np


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

import numpy as np
import multiprocessing as mp
import cv2
import sys

from PIL import Image
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from Preprocessor.pathing import find_path


ROWS = 0
COLUMNS = 1


def binarize_image(image):
    """
    Binarizes a given cv2 image using Otsu's method after Gaussian blurring.
    :param image: cv2 (opencv-python) image.
    :return: binarized cv2 image.
    """

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def count_transistions(row):
    """
    Count the number of transitions between values in an image row.
    :param row: row in binarized image.
    :return: number of transitions in row.
    """

    return len(np.argwhere(np.diff(row)).squeeze())


def find_line_starts(projection):
    """
    Finds the start of line segmentation by taking the center minimal value between peaks.
    :param projection: list of projection values. Each element corresponds to a row in the original data.
    :return: list of row indices where segments start in the data.
    """
    peaks, properties = find_peaks(projection, prominence=1, distance=100)

    minima = []
    for index, peak in enumerate(peaks[:-1]):
        subset = projection[peaks[index]:peaks[index+1]]
        local_minima = np.where(subset == np.amin(subset))[0]
        minima.append(peak + local_minima[int(len(local_minima)/2)])

    return minima


if __name__ == '__main__':

    sys.setrecursionlimit(10 ** 6)
    print("Number of processors: ", mp.cpu_count())

    for i in range(2, 3):

        # Load and binarize image
        print(f"Working on image{i}")
        img = cv2.imread(f"../data/test{i}.jpg", 0)
        img_binarized = binarize_image(img)

        # Compute projection & find line starts.
        data = np.array(img_binarized)
        projection = np.apply_along_axis(count_transistions, COLUMNS, data)
        line_starts = find_line_starts(projection)

        # Generate segment lines per start position
        segment_lines = [find_path(start, data) for start in line_starts]

        # Show projection
        plt.plot(projection)
        plt.plot(line_starts, projection[line_starts], "x")
        plt.title("Ink-paper transition projection with line starts")
        plt.ylabel("Number of transitions")
        plt.xlabel("Pixel row number")
        plt.savefig(f"./results/test{i}-projection-minima.jpg")

        for line in segment_lines:

            # Add (dilated) lines to image
            data[line[:, 0], line[:, 1]] = 0
            data[line[:, 0]-1, line[:, 1]] = 0
            data[line[:, 0]+1, line[:, 1]] = 0

        # Reference lines
        # line_starts += [start + 1 for start in line_starts] + [start - 1 for start in line_starts]
        # data[line_starts] = np.zeros(data.shape[COLUMNS])

        out_segment_image = Image.fromarray(data).save(f"./results/test{i}_segmentlines.jpg")

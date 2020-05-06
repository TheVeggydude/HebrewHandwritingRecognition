import numpy as np
import multiprocessing as mp
import cv2

from matplotlib import pyplot as plt

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


if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())

    img = cv2.imread('../data/test-binarized.jpg', 0)
    img_binarized = binarize_image(img)

    plt.imshow(img_binarized)
    plt.show()

    data = np.array(img_binarized)

    print(np.unique(data))
    projection = np.apply_along_axis(count_transistions, COLUMNS, data)

    plt.plot(projection)
    plt.title("Ink-paper transition projection")
    plt.ylabel("Number of transitions")
    plt.xlabel("Pixel row number")
    plt.savefig("./results/ink-paper-transition-projection.jpg")
    plt.show()

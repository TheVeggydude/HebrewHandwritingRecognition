import numpy as np
import multiprocessing as mp
import cv2

from matplotlib import pyplot as plt


def binarize_image(image):
    """
    Binarizes a given cv2 image using Otsu's method after Gaussian blurring.
    :param image: cv2 (opencv-python) image.
    :return: binarized cv2 image.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())

    img = cv2.imread('../data/test-binarized.jpg', 0)

    result = binarize_image(img)

    plt.imshow(result)
    plt.show()

    data = np.array(result)

    print(np.unique(data))

    plt.hist(data)
    plt.show()

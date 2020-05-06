import numpy as np
import multiprocessing as mp
import time

from PIL import Image
from matplotlib import pyplot

if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())

    data = np.array(Image.open('../data/test-binarized.jpg'))
    rows = data.shape[0]
    columns = data.shape[1]

    print(type(data))
    # summarize shape
    print(data.shape)

    # print data
    print(data)

    # Standard method
    tic = time.perf_counter()

    results = np.apply_along_axis(np.sum, 1, data)
    inverted_results = 255*columns - results

    # Print timing and results
    toc = time.perf_counter()
    print(f"Standard method time: {toc-tic:0.4f} seconds")

    pyplot.plot(inverted_results)
    pyplot.title("Luminosity projection of non-binary image")
    pyplot.xlabel("Pixel 'line' number")
    pyplot.ylabel("Total luminosity")
    pyplot.savefig("./results/inkProjection.jpg")
    pyplot.show()

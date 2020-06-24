import math
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np

from collections import namedtuple
from scipy.signal import find_peaks


# Constants
ROWS = 0
COLUMNS = 1
WINDOW_BUFFER = 10

# Named tuple definitions
Point = namedtuple("Point", ["x", "y"])


class State:
    """
    A state encodes a single state in the search tree. By linking State objects through the parent parameter the path
    can be recreated.
    """
    def __init__(self, coords, parent, f_score, g_score):
        self.coords = coords
        self.parent = parent
        self.priority = f_score
        self.cost = g_score


def compute_heuristic(a, b):
    """
    Computes the heuristic going from coordinates a to coordinates b.
    :param a: Point namedtuple for start coordinates in form (x, y).
    :param b: Point namedtuple for goal coordinates in form (x, y).
    :return: Manhattan distance x 2 from point a to point b.
    """
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def compute_cost(direction):
    """
    Computes the cost for moving in a specified direction, by taking the Euclidean distance between.
    :param direction: Array of length 2 in the form [delta_x, delta_y].
    :return: Floating point cost value for move in specified direction.
    """
    return float(math.sqrt(direction[0] ** 2 + direction[1] ** 2))


def check_in_list(point, lst):
    for elem in lst:
        if elem.coords == point:
            return True, elem
    return False, None


def get_neighbors(state, image):
    moves = [[0, 1], [1, 1], [-1, 1], [1, 0], [-1, 0]]

    # Get a neighbor for each move allowed
    for move in moves:
        neighbor = Point(state.coords.x + move[0], state.coords.y + move[1])

        # Skip if out of bounds
        if neighbor.x >= image.shape[0] or neighbor.x < 0 or neighbor.y >= image.shape[1] or neighbor.y < 0:
            continue

        # Skip if black pixel
        if image[neighbor.x, neighbor.y] == 0:
            continue

        yield neighbor, move


def a_star(row, image):

    # TODO update open to dict or Hash Array Mapped Tries (HAMT) to improve speed
    open_list = []
    visited = set()

    # Determine start and ending coordinates
    starting_coords = Point(row, 0)
    goal_coords = Point(row, image.shape[1] - 1)

    # Create starting node
    h_score = compute_heuristic(starting_coords, goal_coords)
    f_score = h_score
    starting_node = State(starting_coords, None, f_score, 0)

    # Add to open list
    open_list.append(starting_node)

    while True:

        # Get the most optimal node from open list and add it to the closed list
        current = get_lowest_f_node(open_list)
        open_list.remove(current)
        visited.add((current.coords.x, current.coords.y))

        # If goal found, return current node
        if current.coords.x == goal_coords.x and current.coords.y == goal_coords.y:
            return current

        for neighbor, move in get_neighbors(current, image):

            # Skip if already visited
            if (neighbor.x, neighbor.y) in visited:
                continue

            # Compute new scores
            g = compute_cost(move) + current.cost
            h = compute_heuristic(neighbor, goal_coords)
            f = g + h
            new_elem = State(neighbor, current, f, g)

            # Append neighbor to open list
            flag, elem = check_in_list(neighbor, open_list)
            if flag:

                # If neighbor's f score is lower than the one in the open list
                if f < elem.priority:
                    open_list.remove(elem)
                    open_list.append(new_elem)
                else:
                    continue
            else:
                open_list.append(new_elem)


def get_lowest_f_node(open_list):
    """
    Finds and returns the node with the lowest f score in a list.
    :param open_list: List of Node objects to search through.
    :return:
    """
    min_f_score = sys.maxsize
    current = open_list[0]
    for node in open_list:
        if node.priority < min_f_score:
            current = node
            min_f_score = node.priority
    return current


def find_path(row, image):
    max_width = np.shape(image)[COLUMNS] - 1

    # Find speed up variables
    left_margin, right_margin, sub_image = extract_sub_image(image, max_width)

    # Find path to end node
    node = a_star(row, sub_image)

    # Backtrack from ending node to initial node with parent `None`
    prepend_points = np.asarray([[row, column] for column in np.arange(left_margin)])
    append_points = np.asarray([[row, column] for column in np.arange(right_margin + 1, max_width)])
    points = []
    while node is not None:
        points.append((node.coords.x, node.coords.y + left_margin))
        node = node.parent
    points = np.asarray(points)[::-1]  # Flipped so starting node is at arr[0]

    return np.concatenate((prepend_points, points, append_points))


def extract_sub_image(image, max_width):
    relevant_columns = [0, max_width]

    # Find first column that shows a pixel.
    for column in range(image.shape[COLUMNS]):
        transitions = count_transitions(image[:, column])
        if transitions > 0:
            relevant_columns[0] = column
            break

    # Find last column that shows a pixel.
    for column in range(image.shape[COLUMNS] - 1, 0, -1):
        transitions = count_transitions(image[:, column])
        if transitions > 0:
            relevant_columns[1] = column
            break

    # Compute margins
    left_margin = relevant_columns[0]
    left_margin = left_margin - WINDOW_BUFFER if left_margin - WINDOW_BUFFER > 0 else 0

    right_margin = relevant_columns[1]
    right_margin = right_margin + WINDOW_BUFFER if right_margin + WINDOW_BUFFER < max_width else max_width

    return left_margin, right_margin, image[:, left_margin: right_margin]


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

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def count_transitions(row):
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

    for i in range(0, 3):
        # Load and binarize image
        print(f"Working on test image {i}")
        img = cv2.imread(f"../data/test{i}.jpg", 0)
        img_binarized = binarize_image(img)

        # Compute projection & find line starts.
        img_arr = np.array(img_binarized)
        print(f"Image shape: {img_arr.shape}")
        projection = np.apply_along_axis(count_transitions, COLUMNS, img_arr)
        line_starts = find_line_starts(projection)

        for index, start in enumerate(line_starts):
            print(f"Finding path for line at row {start} ({index+1}/{len(line_starts)}).")
            path = find_path(start, img_arr)

            # TODO verify path integrity
            plt.plot(path[:, COLUMNS], path[:, ROWS])
        plt.imshow(img_arr, 'gray')
        plt.show()

        # plt.imshow(extracted_image, 'gray', vmin=0)
        # plt.show()

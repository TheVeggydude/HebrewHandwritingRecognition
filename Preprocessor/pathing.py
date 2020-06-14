import numpy as np

from queue import PriorityQueue
from PIL import Image
from matplotlib import pyplot as plt

ROWS = 0
COLUMNS = 1


class State:
    """Class to save all the characteristics of a pixel"""
    def __init__(self, coords, parent, cost=0, priority=0):
        self.parent = parent
        self.coords = coords
        self.cost = cost
        self.priority = priority

    # Function used to compare two states for the priority queue
    def __lt__(self, other):
        return self.priority < other.priority


def list_path(state, column_offset=0):
    """
    Generates the list of the coordinates in the path leading up to the current State.
    :param column_offset:
    :param state: State object that you want the path to generated for.
    :return: List of tuples, each tuple is a coordinate.
    """
    coords = [(state.coords[0], state.coords[1] + column_offset)]
    if state.parent is None:  # Base case, so return coords
        return coords

    # Recursive case, so return using recursive call
    return list_path(state.parent, column_offset) + coords


def get_neighbors(state, data, goal):
    """
    Iteration method that yields all legal neighboring coordinates.
    :param state: State object to get neighbours for.
    :param data: Binarized image data.
    :return: Neighboring coordinates and corresponding move cost per iteration.
    """
    max_values = np.shape(data)

    for d_row in [0, 1, -1]:
        new_row = state.coords[0] + d_row
        if new_row >= max_values[0] or new_row < 0:  # Don't look for out of bounds values
            continue

        if abs(new_row - goal[0]) >= 100:  # Don't stray more than 20 rows from target line
            continue

        columns = [0, 1] if d_row != 0 else [1]   # Don't check for zero-offset "neighbors"
        for d_column in columns:
            new_column = state.coords[1] + d_column
            if new_column >= max_values[1] or new_column < 0:  # Don't look for out of bounds values
                continue

            if data[new_row, new_column] == 0:  # Don't go through black pixels
                continue

            yield new_row, new_column


def get_move_cost(coords, neighbor, pixel, goal):
    """
    Cost = 1
    :param coords:
    :param neighbor:
    :param pixel:
    :return:
    """
    cost = 1
    return cost


def compute_heuristic(coords, goal):
    """
    Computes the heuristic value between a set of coordinates and the goal coordinates using Chebyshev distance.
    :param coords: Tuple of starting coordinates.
    :param goal: Tuple of goal coordinates.
    :return: Chebyshev distance.
    """
    dy = abs(coords[0] - goal[0])
    dx = abs(coords[1] - goal[1])
    return max(dx, dy)


def count_transitions(column):
    """
    Count the number of transitions between values in an image row.
    :param column: row in binarized image.
    :return: number of transitions in row.
    """

    return len(np.argwhere(np.diff(column)).squeeze())


def find_path(row, image):
    """
    Generates a line-path from the starting point using the A* path planning method.
    :param row: Row number for which to start a segment at.
    :param image: Binarized image data in NumPy array.
    :return: 2D numpy array of coordinates describing the optimal path.
    """

    print(f"Finding path for row {row}...")
    window_buffer = 10
    max_width = np.shape(image)[COLUMNS] - 1

    relevant_columns = [0, max_width]

    # Find first column that shows a pixel.
    for column in range(image.shape[COLUMNS]):
        transitions = count_transitions(image[:, column])
        if transitions > 0:
            relevant_columns[0] = column
            break

    # Find last column that shows a pixel.
    for column in range(image.shape[COLUMNS]-1, 0, -1):
        transitions = count_transitions(image[:, column])
        if transitions > 0:
            relevant_columns[1] = column
            break

    # Take subset of image as current 'maze'
    left_margin = relevant_columns[0]
    left_margin = left_margin - window_buffer if left_margin - window_buffer > 0 else 0

    right_margin = relevant_columns[1]
    right_margin = right_margin + window_buffer if right_margin + window_buffer < max_width else max_width

    sub_image = image[:, left_margin: right_margin]

    # Initiate straight line as path up to area of interest
    path = np.asarray([[row, column] for column in np.arange(left_margin)])
    start = (row, 0)
    goal = (row, np.shape(sub_image)[COLUMNS]-1)

    # Create starting state and add to queue
    state = State(start, None)
    queue = PriorityQueue()
    queue.put(state)

    # Keep track of visited list
    visited = set()

    # Loop until all options are used up
    while not queue.empty():

        # Get next state and add to visited set
        state = queue.get()
        visited.add(state.coords)

        # Check for goal state
        if state.coords[0] == goal[0] and state.coords[1] == goal[1]:
            app_path = np.asarray([[row, column] for column in np.arange(right_margin + 1, max_width)])
            segment = np.asarray(list_path(state, left_margin))
            path = np.concatenate((path, segment))  # append the found path
            path = np.concatenate((path, app_path))
            return path

        # Check every possible direction
        for neighbor in get_neighbors(state, sub_image, goal):

            # Only handle unvisited neighbours
            if neighbor not in visited:
                new_cost = state.cost + get_move_cost(state.coords, neighbor, sub_image[neighbor[0], neighbor[1]], goal)
                priority = new_cost + compute_heuristic(neighbor, goal)
                new_state = State(neighbor, state, new_cost, priority)
                queue.put(new_state)


if __name__ == '__main__':
    """
    For testing purposes. Run the path-finding on a simple maze.
    """

    data = np.full((10, 10), 255)
    data[1:4, 2] = 0

    path = find_path(2, data)
    print("Path found: ")
    data[path[:, 0], path[:, 1]] = -1
    print(data)

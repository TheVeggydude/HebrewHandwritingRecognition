import numpy as np

from queue import PriorityQueue
from PIL import Image

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


def list_path(state):
    """
    Generates the list of the coordinates in the path leading up to the current State.
    :param state: State object that you want the path to generated for.
    :return: List of tuples, each tuple is a coordinate.
    """
    if state.parent is None:  # Base case, so return coords
        return [state.coords]

    # Recursive case, so return using recursive call
    return list_path(state.parent) + [state.coords]


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
    For now, just give high cost to moving through black pixels.
    :param coords:
    :param neighbor:
    :param pixel:
    :return:
    """
    cost = 1  # + abs(coords[0] - goal[0])**2
    # cost += 1000 if neighbor[0] != goal_height else 0

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


def find_path(row, image):
    """
    Generates a line-path from the starting point using the A* path planning method.
    :param row: Row number for which to start a segment at.
    :param image: Binarized image data in NumPy array.
    :return: 2D numpy array of coordinates describing the optimal path.
    """

    print(f"Finding path for row {row}...")
    limits = [50, 100]

    # Start and goal coordinates
    barriers = np.where(image[row, :] == 0)[0]
    start_column = barriers[0] - limits[0] if len(barriers) > 0 and barriers[0] > limits[0] else 0

    img_width = np.shape(image)[COLUMNS]-1
    goal_column = barriers[-1] + limits[0] if len(barriers) > 1 and barriers[-1] + limits[0] < img_width else img_width

    image = image[row-100:row+101, start_column:goal_column]
    sub_shape = np.shape(image)

    start = (int(sub_shape[0] / 2), 0)
    goal = (int(sub_shape[0] / 2), sub_shape[1]-1)
    print(start, goal)

    # Create starting state and add to queue
    state = State(start, None)
    queue = PriorityQueue()
    queue.put(state)

    # Keep track of visited list
    visited = set()
    path = None

    # Loop until all options are used up
    while not queue.empty():

        if len(visited) % 10 == 0 and len(visited) != 0:
            print(len(visited))

        # Get next state and add to visited set
        state = queue.get()
        visited.add(state.coords)

        # Check for goal state
        if state.coords[0] == goal[0] and state.coords[1] == goal[1]:
            path = np.asarray(list_path(state))  # returns the array of coordinates
            break

        # Check every possible direction
        for neighbor in get_neighbors(state, image, goal):

            # Only handle unvisited neighbours
            if neighbor not in visited:
                new_cost = state.cost + get_move_cost(state.coords, neighbor, image[neighbor[0], neighbor[1]], goal)
                priority = new_cost + compute_heuristic(neighbor, goal)
                new_state = State(neighbor, state, new_cost, priority)
                queue.put(new_state)

    image[path[:, 0], path[:, 1]] = 0
    image[path[:, 0] - 1, path[:, 1]] = 0
    image[path[:, 0] + 1, path[:, 1]] = 0

    test_image = Image.fromarray(image).save(f"./results/fuck.jpg")

    print(path)
    exit(-1)
    return path


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

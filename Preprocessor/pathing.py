import numpy as np

from queue import PriorityQueue

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


def get_neighbors(state, data):
    """
    Iteration method that yields all legal neighboring coordinates.
    :param state: State object to get neighbours for.
    :param data: Binarized image data.
    :return: Neighboring coordinates and corresponding move cost per iteration.
    """
    max_values = np.shape(data)

    for d_row in [1, 0, -1]:
        new_row = state.coords[0] + d_row
        if new_row >= max_values[0] or new_row < 0:  # Don't look for out of bounds values
            continue

        columns = [1, 0] if d_row != 0 else [1]  # Don't check for zero-offset "neighbors"
        for d_column in columns:
            new_column = state.coords[1] + d_column
            if new_column >= max_values[1] or new_column < 0:  # Don't look for out of bounds values
                continue

            if data[new_row, new_column] == 0:  # Don't go through black pixels
                continue

            yield new_row, new_column


def get_move_cost(coords, neighbor, pixel, goal_height):
    """
    For now, just give high cost to moving through black pixels.
    :param coords:
    :param neighbor:
    :param pixel:
    :return:
    """
    cost = 1
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

    # Start and goal coordinates
    start = (row, 0)
    goal = (row, np.shape(image)[COLUMNS]-1)

    # Create starting state and add to queue
    state = State(start, None)
    queue = PriorityQueue()
    queue.put(state)

    # Keep track of visited list
    visited = set()

    # Loop until all options are used up
    while not queue.empty():

        if len(visited) % 100 == 0:
            print(len(visited))

        # Get next state and add to visited set
        state = queue.get()
        visited.add(state.coords)

        # Check for goal state
        if state.coords[0] == goal[0] and state.coords[1] == goal[1]:
            return np.asarray(list_path(state))  # returns the array of coordinates

        # Check every possible direction
        for neighbor in get_neighbors(state, image):

            # Only handle unvisited neighbours
            if neighbor not in visited:
                new_cost = state.cost + get_move_cost(state.coords, neighbor, image[neighbor[0], neighbor[1]], goal[0])
                priority = new_cost + compute_heuristic(neighbor, goal)
                new_state = State(neighbor, state, new_cost, priority)
                queue.put(new_state)

    # Path not found, throw temp error
    raise ValueError
    # return None


if __name__ == '__main__':
    """
    For testing purposes. Run the path-finding on a simple maze.
    """

    data = np.full((5, 5), 255)
    data[1:4, 2] = 0

    path = find_path(2, data)
    print("Path found: ")
    data[path[:, 0], path[:, 1]] = -1
    print(data)

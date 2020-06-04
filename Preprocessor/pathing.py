import numpy as np

from queue import PriorityQueue

ROWS = 0
COLUMNS = 1


class State:
    """Class to save all the characteristics of a pixel"""
    def __init__(self, coords, parent, cost=0, prio=0):
        self.parent = parent
        self.coords = coords
        self.cost = cost
        self.prio = prio
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def print_path_helper(self):
        if self.parent is not None:
            self.parent.print_path_helper()
            string = str(self.parent.coords) + " -> "
            string += str(self.coords)
            string += " cost: "+str(self.cost)
            print(string)

    def print_path(self):
        self.print_path_helper()
        print()

    # Function used to compare two states for the priority queue
    def __lt__(self, other):
        return self.prio < other.prio


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


def get_neighbors(state):
    """
    Iteration method that yields all legal neighboring coordinates.
    :param state: State object to get neighbours for.
    :return: Neighboring state per iteration.
    """

    for d_row in [1, 0, -1]:
        for d_column in [1, 0]:
            yield state.coords[0] + d_row, state.coords[1] + d_column



def find_path(row, image):
    """
    Generates a line-path from the starting point using the A* path planning method.
    :param row: Row number for which to start a segment at.
    :param image: Binarized image data in NumPy array.
    :return: List of coordinate tuples describing the optimal path.
    """

    # Start and goal coordinates
    start = (row, 0)
    goal = (row, np.shape(image)[COLUMNS])

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
        if state.coords == goal:
            return list_path(state)  # returns the list of coordinates

        # Check every possible direction
        for neighbor in get_neighbors(state):
            # TODO make the actual moves.
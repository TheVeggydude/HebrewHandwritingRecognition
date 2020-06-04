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


def get_neighbors(state):
    """
    Iteration method that yields all legal neighboring coordinates.
    :param state: State object to get neighbours for.
    :return: Neighboring coordinates and corresponding move cost per iteration.
    """

    for d_row in [1, 0, -1]:
        for d_column in [1, 0]:
            yield state.coords[0] + d_row, state.coords[1] + d_column


def get_move_cost(coords, neighbor, pixel, goal):
    """
    For now, just give high cost to moving through black pixels.
    :param coords:
    :param neighbor:
    :param pixel:
    :return:
    """
    cost = 100 if not pixel else 1
    cost += 20 if coords[0] != goal[0] else 0

    return cost
    # Make make vertical move > diagonal move > horizontal move to show that straight lines are preferred. Moreover,
    # make moves going back to the starting line less costly.


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

        if len(visited) % 100 == 0:
            print(len(visited))

        # Get next state and add to visited set
        state = queue.get()
        visited.add(state.coords)

        # Check for goal state
        if state.coords == goal:
            return list_path(state)  # returns the list of coordinates

        # Check every possible direction
        for neighbor in get_neighbors(state):

            # Only handle unvisited neighbours
            if neighbor not in visited:
                new_cost = state.cost + get_move_cost(state.coords, neighbor, image[neighbor[0], neighbor[1]], goal)
                priority = new_cost + compute_heuristic(neighbor, goal)
                new_state = State(neighbor, state, new_cost, priority)
                queue.put(new_state)

                print(f"{state.coords} --> {neighbor}: cost: {new_cost}, priority: {priority}")

        exit()

    # Path not found, throw temp error
    raise ValueError
    # return None

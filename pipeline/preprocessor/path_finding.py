import sys
import numpy as np

from .utils import extract_sub_image, COLUMNS, ROWS, MOVES, Point


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
    :return: Euclidean distance x 2 from point a to point b.
    """
    return max(abs(a.x - b.x), abs(a.y - b.y))


def compute_cost(move):
    """
    Computes the cost for moving in a specified direction, by taking the Manhattan distance between.
    :param move: Array of length 2 in the form [delta_x, delta_y].
    :return: Cost value for move in specified direction.
    """
    return move[0] ** 2 + move[1] ** 2


def search_for(coords, lst):
    """
    Checks a list of States for a state with a particular set of coordinates and returns it.
    :param coords: Point namedtuple containing the coordinates to search for.
    :param lst: List of State objects to search through.
    :return: Boolean showing if the coordinates were found plus the corresponding State in a tuple.
    """
    for elem in lst:
        if elem.coords == coords:
            return True, elem
    return False, None


def get_neighbors(state, image):
    """
    Generator function that generates the possible neighbors of a state given the image.
    :param state: State object representing the current point in the image.
    :param image: 2D NumPy array describing the image in either black (= 0) or white (= 255) pixels.
    :return: A list of pairs of form Point, move describing the new coordinates and the move that got there.
    """

    # Get a neighbor for each move allowed
    for move in MOVES:
        neighbor = Point(state.coords.x + move[0], state.coords.y + move[1])

        # Skip if out of bounds
        if neighbor.x >= image.shape[0] or neighbor.x < 0 or neighbor.y >= image.shape[1] or neighbor.y < 0:
            continue

        # Skip if black pixel
        if image[neighbor.x, neighbor.y] == 0:
            continue

        yield neighbor, move


def get_lowest_prio_state(open_list):
    """
    Finds and returns the node with the lowest f score in a list.
    :param open_list: List of State objects to search through.
    :return:
    """
    minimal = sys.maxsize
    current = open_list[0]
    for node in open_list:
        if node.priority < minimal:
            current = node
            minimal = node.priority
    return current


def a_star(row, maze):
    open_list = []
    visited = set()

    # Determine start and ending coordinates
    starting_coords = Point(row, 0)
    goal_coords = Point(row, maze.shape[1] - 1)

    # Create starting node
    h_score = compute_heuristic(starting_coords, goal_coords)
    f_score = h_score
    starting_node = State(starting_coords, None, f_score, 0)

    # Add to open list
    open_list.append(starting_node)

    while True:

        # No path found, so
        if not open_list:
            return None

        # Get the most optimal node from open list and add it to the closed list
        current = get_lowest_prio_state(open_list)
        open_list.remove(current)
        visited.add((current.coords.x, current.coords.y))

        # If goal found, return current node
        if current.coords.x == goal_coords.x and current.coords.y == goal_coords.y:
            return current

        for neighbor, move in get_neighbors(current, maze):

            # Skip if already visited
            if (neighbor.x, neighbor.y) in visited:
                continue

            # Compute new scores
            g = compute_cost(move) + current.cost
            h = compute_heuristic(neighbor, goal_coords)
            f = g + h
            new_elem = State(neighbor, current, f, g)

            # Append neighbor to open list
            flag, elem = search_for(neighbor, open_list)
            if flag:

                # If neighbor's f score is lower than the one in the open list
                if f < elem.priority:
                    open_list.remove(elem)
                    open_list.append(new_elem)
                else:
                    continue
            else:
                open_list.append(new_elem)


def find_path(row, peaks, image):
    """
    Finds a path between left most edge and the corresponding right edge of the image at a specific row height. Performs
    some optimizations on the image array to decrease computation times.
    :param row: Integer describing the row in the image array to search along.
    :param peaks: List of form [top_row, bottom_row] in between which the path must be found.
    :param image: 2D NumPy array with the image data.
    :return: List of tuples describing the coordinates in the path.
    """
    max_width = np.shape(image)[COLUMNS] - 1

    # Find speed up variables
    left_margin, right_margin, row_offset, sub_image = extract_sub_image(image, peaks)

    # Find path to end node
    node = a_star(row-row_offset, sub_image)

    # In case of no path found, return none.
    if not node:
        return None

    # Backtrack from ending node to initial node with parent `None`
    prepend_points = np.asarray([[row, column] for column in np.arange(left_margin)])
    append_points = np.asarray([[row, column] for column in np.arange(right_margin, max_width+1)])

    points = []
    while node is not None:
        points.append((node.coords.x + row_offset, node.coords.y + left_margin))
        node = node.parent
    points = np.asarray(points)[::-1]  # Flipped so starting node is at arr[0]

    # Validate used point lists
    used_lists = []
    if prepend_points != []:
        used_lists.append(prepend_points)

    if points != []:
        used_lists.append(points)

    if append_points != []:
        used_lists.append(append_points)

    return np.concatenate(tuple(used_lists))

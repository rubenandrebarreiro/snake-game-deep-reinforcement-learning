from numpy.linalg import norm

from numpy import array

from math import atan2

from math import pi

from game.utils.plotting_helper import plot_board

from game.others.parameters_arguments import AVAILABLE_DISTANCES_LIST

BORDER_COLOR = [0.5, 0.5, 0.5]

APPLE_COLOR = [0.0, 1.0, 0.0]

SNAKE_HEAD_COLOR = [1.0, 1.0, 1.0]

SNAKE_BODY_COLOR = [1.0, 0.0, 0.0]


def get_distance_from_snake_to_apple(snake_head_position, apple_position,
                                     distance=AVAILABLE_DISTANCES_LIST[1]):

    # It was chosen the Norm Euclidean Distance for the Heuristics
    if distance == AVAILABLE_DISTANCES_LIST[0]:
        return get_euclidean_norm_distance_from_snake_to_apple(snake_head_position, apple_position)

    # It was chosen the Manhattan Distance for the Heuristics
    elif distance == AVAILABLE_DISTANCES_LIST[1]:
        return get_manhattan_distance_from_snake_to_apple(snake_head_position, apple_position)


def get_euclidean_norm_distance_from_snake_to_apple(snake_head_position, apple_position):
    return norm(array(apple_position) - array(snake_head_position))


def get_manhattan_distance_from_snake_to_apple(snake_head_position, apple_position):
    return sum(abs(coordinate_1 - coordinate_2) for coordinate_1, coordinate_2 in
               zip(array(apple_position), array(snake_head_position)))


def check_suicide_against_itself(snake_head_position, snake_body_positions):
    return snake_head_position in snake_body_positions


def check_suicide_against_borders(snake_head_position, board_state_matrix, board_state_border):

    board_height = board_state_matrix.shape[0] - (2 * board_state_border)
    board_width = board_state_matrix.shape[1] - (2 * board_state_border)

    return snake_head_position[0] >= board_width or snake_head_position[0] <= 0 or \
        snake_head_position[1] >= board_height or snake_head_position[1] <= 0


def check_apple_eaten(snake_head_position, apple_position):
    return apple_position == snake_head_position


def is_direction_dangerous(snake_head_position, snake_body_positions, current_direction_vector,
                           board_state_matrix, board_state_border):

    next_snake_head_position = snake_head_position + current_direction_vector

    return check_suicide_against_itself(tuple(next_snake_head_position), snake_body_positions) or \
        check_suicide_against_borders(next_snake_head_position, board_state_matrix, board_state_border)


def get_dangerous_directions(snake_head_position, snake_body_positions, board_state_matrix, board_state_border):

    current_direction_vector = array(snake_head_position) - array(snake_body_positions[0])

    left_direction_vector = array([-current_direction_vector[1], current_direction_vector[0]])
    right_direction_vector = array([current_direction_vector[1], -current_direction_vector[0]])

    is_front_dangerous = is_direction_dangerous(snake_head_position, snake_body_positions,
                                                current_direction_vector, board_state_matrix, board_state_border)

    is_left_dangerous = is_direction_dangerous(snake_head_position, snake_body_positions,
                                               left_direction_vector, board_state_matrix, board_state_border)

    is_right_dangerous = is_direction_dangerous(snake_head_position, snake_body_positions,
                                                right_direction_vector, board_state_matrix, board_state_border)

    return current_direction_vector, is_front_dangerous, is_left_dangerous, is_right_dangerous


def get_angle_with_apple(snake_head_position, snake_body_positions, apple_position):

    apple_direction_vector = array(apple_position) - array(snake_head_position)
    snake_direction_vector = array(snake_head_position) - array(snake_body_positions[0])

    norm_of_apple_direction_vector = norm(apple_direction_vector)
    norm_of_snake_direction_vector = norm(snake_direction_vector)

    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 1

    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 1

    apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector

    angle_with_apple = atan2(apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] -
                             apple_direction_vector_normalized[0] * snake_direction_vector_normalized[1],
                             apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] +
                             apple_direction_vector_normalized[0] * snake_direction_vector_normalized[0]) / pi

    return angle_with_apple, snake_direction_vector,\
        apple_direction_vector_normalized, snake_direction_vector_normalized


def generate_direction_to_apple(angle_with_apple):

    if angle_with_apple > 0:
        direction = -1
    elif angle_with_apple < 0:
        direction = 1
    else:
        direction = 0

    return direction


def print_board_status(borders_positions, apples_positions, snake_head_position, snake_body_positions):

    print("\n")
    print("\n")

    print("Borders' Positions:")
    print(*borders_positions, sep="\n")

    print("\n")

    print("Apples' Positions:")
    print(*apples_positions, sep="\n")

    print("\n")

    print("Snake's Head Position:")
    print(snake_head_position, sep="\n")

    print("\n")

    print("Snake's Body Positions:")
    print(*snake_body_positions, sep="\n")

    print("\n")

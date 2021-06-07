from game.snake_game import SnakeGame

from numpy.linalg import norm

from numpy import array

from math import atan2

from math import pi

from game.utils.plotting_helper import plot_board

BORDER_COLOR = [0.5, 0.5, 0.5]

APPLE_COLOR = [0.0, 1.0, 0.0]

SNAKE_HEAD_COLOR = [1.0, 1.0, 1.0]

SNAKE_BODY_COLOR = [1.0, 0.0, 0.0]


def get_initial_positions(board_state_matrix):

    board_height = board_state_matrix.shape[0]
    board_width = board_state_matrix.shape[1]

    borders_positions = []
    apples_positions = []
    snake_head_position = None
    snake_body_positions = []

    for current_row in range(board_height):

        for current_column in range(board_width):

            current_color = board_state_matrix[current_row, current_column]

            # The current position of the Board is part of the Border
            if (current_color == BORDER_COLOR).all():
                borders_positions.append([current_row, current_column])

            # The current position of the Board is an Apple
            if (current_color == APPLE_COLOR).all():
                apples_positions.append([current_row, current_column])

            # The current position of the Board is the Snake's Head
            if (current_color == SNAKE_HEAD_COLOR).all():
                snake_head_position = [current_row, current_column]

            # The current position of the Board is part of the Snake's Body
            if (current_color == SNAKE_BODY_COLOR).all():
                snake_body_positions.append([current_row, current_column])

    return borders_positions, apples_positions, snake_head_position, snake_body_positions


def get_euclidean_norm_distance_from_snake_to_apple(snake_head_position, apple_position):
    return norm(array(apple_position) - array(snake_head_position))


def get_manhattan_distance_from_snake_to_apple(snake_head_position, apple_position):
    return sum(abs(coordinate_1 - coordinate_2) for coordinate_1, coordinate_2 in
               zip(array(apple_position), array(snake_head_position)))


def check_suicide_against_itself(snake_head_position, snake_body_positions):
    return snake_head_position in snake_body_positions


def check_suicide_against_borders(snake_head_position, board_state_matrix):

    board_height = board_state_matrix.shape[0]
    board_width = board_state_matrix.shape[1]

    return snake_head_position[0] >= board_width or snake_head_position[0] < 0 or \
        snake_head_position[1] >= board_height or snake_head_position[1] < 0


def check_apple_eaten(snake_head_position, apple_position):
    return apple_position == snake_head_position


def is_direction_dangerous(snake_head_position, snake_body_positions, current_direction_vector, board_state_matrix):

    next_snake_head_positions = snake_head_position + current_direction_vector

    return check_suicide_against_itself(next_snake_head_positions.to_list(), snake_body_positions) or \
        check_suicide_against_borders(next_snake_head_positions, board_state_matrix)


def get_dangerous_directions(snake_head_position, snake_body_positions, board_state_matrix):

    current_direction_vector = array(snake_head_position) - array(snake_body_positions[0])
    left_direction_vector = array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = array([-current_direction_vector[1], current_direction_vector[0]])

    is_front_dangerous = is_direction_dangerous(snake_head_position, snake_body_positions,
                                                current_direction_vector, board_state_matrix)

    is_left_dangerous = is_direction_dangerous(snake_head_position, snake_body_positions,
                                               left_direction_vector, board_state_matrix)

    is_right_dangerous = is_direction_dangerous(snake_head_position, snake_body_positions,
                                                right_direction_vector, board_state_matrix)

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


def get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, last_direction):

    current_direction_vector = array(snake_head_position) - array(snake_body_positions[0])
    left_direction_vector = array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = array([-current_direction_vector[1], current_direction_vector[0]])

    new_direction_vector = current_direction_vector

    if last_direction == -1:
        new_direction_vector = left_direction_vector

    if last_direction == 1:
        new_direction_vector = right_direction_vector

    button_direction = generate_button_direction(new_direction_vector)

    return last_direction, button_direction


def generate_button_direction(new_direction_vector):

    button_direction = 0

    if new_direction_vector.tolist() == [-1, 0]:
        button_direction = 0

    elif new_direction_vector.tolist() == [1, 0]:
        button_direction = 1

    elif new_direction_vector.tolist() == [0, 1]:
        button_direction = 2

    elif new_direction_vector.tolist() == [0, -1]:
        button_direction = 3

    return button_direction


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

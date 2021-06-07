#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Deep Learning - Assignment #2:
- Snake Game - Deep Reinforcement Learning

Integrated Master of Computer Science and Engineering

NOVA School of Science and Technology,
NOVA University of Lisbon - 2020/2021

Authors:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

Instructor(s):
- Ludwig Krippahl (ludi@fct.unl.pt)
- Claudia Soares (claudia.soares@fct.unl.pt)

Snake Game Module for the the Project
(given by the Instructor(s))

"""

# Import Python's Modules, Libraries and Packages

# From the Game.Snake_Game Module,
# import the Snake Game object
from game.snake_game import SnakeGame

# From the Game.Utils.Board_State_Utils Module,
# import the function to get the Distance to an Apple
from game.utils.board_state_utils import get_distance_from_snake_to_apple

# From the Game.Utils.Board_State_Utils Module,
# import the function to get the Direction Vector to an Apple
from game.utils.board_state_utils import get_direction_vector_to_the_apple

# From the Game.Utils.Board_State_Utils Module,
# import the function to get the Initial Positions of the Game
from game.utils.board_state_utils import get_initial_positions

# From the Game.Utils.Board_State_Utils Module,
# import the function to get the Angle with an Apple
from game.utils.board_state_utils import get_angle_with_apple

# From the Game.Utils.Board_State_Utils Module,
# import the function to generate Direction to an Apple
from game.utils.board_state_utils import generate_direction_to_apple

# From the Game.Utils.Board_State_Utils Module,
# import the function to get Dangerous Directions
from game.utils.board_state_utils import get_dangerous_directions

# From the Game.Others.Parameters_Arguments,
# import the Distance being currently used
from game.others.parameters_arguments import DISTANCE_USED


def generate_training_data(width, height, food_amount=1, border=0, grass_growth=0, max_grass=0):

    training_data_xs = []
    training_data_ys = []

    num_training_games = 1000
    num_steps_per_game = 2000

    snake_game = SnakeGame(width, height, food_amount, border, grass_growth, max_grass)

    for _ in range(num_training_games):

        board_state, reward, done, score_dictionary = snake_game.reset()

        borders_positions, apples_positions, snake_head_position, snake_body_positions = \
            get_initial_positions(board_state)

        previous_distance_from_snake_to_apple = \
            get_distance_from_snake_to_apple(snake_head_position, apples_positions[0], DISTANCE_USED)

        for _ in range(num_steps_per_game):

            angle_with_apple, snake_direction_vector, \
                apple_direction_vector_normalized, snake_direction_vector_normalized = \
                get_angle_with_apple(snake_head_position, snake_body_positions, apples_positions[0])

            generated_direction, button_direction = \
                generate_direction_to_apple(snake_head_position, snake_body_positions, angle_with_apple)

            current_direction_vector, is_front_dangerous, is_left_dangerous, is_right_dangerous = \
                get_dangerous_directions(snake_head_position, snake_body_positions, board_state)

            direction, button_direction, training_data_ys = \
                generate_training_data_ys(snake_head_position, snake_body_positions,
                                          button_direction, generated_direction, training_data_ys,
                                          is_front_dangerous, is_left_dangerous, is_right_dangerous)

            if is_front_dangerous and is_left_dangerous and is_right_dangerous:
                break

            training_data_xs.append([is_left_dangerous, is_front_dangerous, is_right_dangerous,
                                     apple_direction_vector_normalized[0], snake_direction_vector_normalized[0],
                                     apple_direction_vector_normalized[1], snake_direction_vector_normalized[1]])

    return training_data_xs, training_data_ys


def generate_training_data_ys(snake_head_position, snake_body_positions,
                              button_direction, generated_direction, training_data_ys,
                              is_front_dangerous, is_left_dangerous, is_right_dangerous):

    direction = 0

    if generated_direction == -1:
        if is_left_dangerous:
            if is_front_dangerous and not is_right_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, 1)
                training_data_ys.append([0, 0, 1])
            elif not is_front_dangerous and is_right_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, 0)
                training_data_ys.append([0, 1, 0])
            elif not is_front_dangerous and not is_right_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, 1)
                training_data_ys.append([0, 0, 1])
        else:
            training_data_ys.append([1, 0, 0])

    elif generated_direction == 0:
        if is_front_dangerous:
            if is_left_dangerous and not is_right_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, 1)
                training_data_ys.append([0, 0, 1])
            elif not is_left_dangerous and is_right_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, -1)
                training_data_ys.append([1, 0, 0])
            elif not is_left_dangerous and not is_right_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, 1)
                training_data_ys.append([0, 0, 1])

        else:
            training_data_ys.append([0, 1, 0])

    elif generated_direction == 1:
        if is_right_dangerous:
            if is_left_dangerous and not is_front_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, 0)
                training_data_ys.append([0, 1, 0])
            elif not is_left_dangerous and is_front_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, -1)
                training_data_ys.append([1, 0, 0])
            elif not is_left_dangerous and is_front_dangerous:
                direction, button_direction = \
                    get_direction_vector_to_the_apple(snake_head_position, snake_body_positions, -1)
                training_data_ys.append([1, 0, 0])
        else:
            training_data_ys.append([0, 0, 1])

    return direction, button_direction, training_data_ys

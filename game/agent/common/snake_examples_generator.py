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


from game.utils.plotting_helper import plot_board

from game.snake_game import SnakeGame

# From the Game.Utils.Board_State_Utils Module,
# import the function to get the Angle with an Apple
from game.utils.board_state_utils import get_angle_with_apple

# From the Game.Utils.Board_State_Utils Module,
# import the function to generate Direction to an Apple
from game.utils.board_state_utils import generate_direction_to_apple

# From the Game.Utils.Board_State_Utils Module,
# import the function to get Dangerous Directions
from game.utils.board_state_utils import get_dangerous_directions


def generate_examples_data_for_replay_memory(width, height, food_amount=1, border=0, grass_growth=0, max_grass=0):

    examples_data_for_replay_memory = []

    num_training_games = 100
    num_steps_per_game = 1000

    snake_game = SnakeGame(width=width, height=height, food_amount=food_amount, border=border,
                           grass_growth=grass_growth, max_grass=max_grass)

    for current_num_training_game in range(num_training_games):

        print("Starting the Generation of the Data for the Training Game #{}...\n"
              .format((current_num_training_game + 1)))

        board_state, reward, done, score_dictionary = snake_game.reset()

        #  TODO
        """
        apples_positions = snake_game.apples
        print("maca at ", apples_positions[0])
        print("snake: ", snake_game.snake)
        """

        # Retrieve the name of the Action taken
        #action_name = {-1: "Turn Left", 0: "Straight Ahead", 1: "Turn Right"}

        # Plot the Board of the Snake Game for the initial state
        #plot_board("{}.png".format(0), board_state, "Start")

        for current_num_step in range(num_steps_per_game):

            print("Generating the Data for the Step #{}..."
                  .format((current_num_step + 1)))

            snake_head_position = snake_game.snake[0]
            snake_body_positions = snake_game.snake[1:]
            apples_positions = snake_game.apples

            angle_with_apple, snake_direction_vector, \
                apple_direction_vector_normalized, snake_direction_vector_normalized = \
                get_angle_with_apple(snake_head_position, snake_body_positions, apples_positions[0])

            generated_direction = generate_direction_to_apple(angle_with_apple)

            current_direction_vector, is_front_dangerous, is_left_dangerous, is_right_dangerous = \
                get_dangerous_directions(snake_head_position, snake_body_positions, board_state, snake_game.border)

            direction = update_generated_direction(generated_direction,
                                                   is_front_dangerous, is_left_dangerous, is_right_dangerous)

            # Retrieve the old observation made by the Snake Agent
            snake_old_observation = snake_game.board_state()

            board_state, reward, done, score_dictionary = snake_game.play_step(direction)

            snake_new_observation = board_state

            snake_action = direction

            examples_data_for_replay_memory.append([snake_old_observation, snake_action, reward,
                                                    snake_new_observation, done])

            if done:

                # Retrieve the current Score from the Dictionary of the Score
                current_score = score_dictionary["Score"]

                print("Game Over!!! Final Score: {}".format(current_score))
                print("\n")

                break

            # Plot the Board of the Game for the Action taken
            #plot_board("{}.png".format(current_num_step+1), board_state, action_name[direction])

            # Retrieve the current Score from the Dictionary of the Score
            current_score = score_dictionary["Score"]

            print("Current Score: {}".format(current_score))
            print("\n")

    return examples_data_for_replay_memory


def update_generated_direction(generated_direction, is_front_dangerous, is_left_dangerous, is_right_dangerous):

    if generated_direction == -1:
        if is_left_dangerous:
            if is_front_dangerous and not is_right_dangerous:
                generated_direction = 1
            elif not is_front_dangerous and is_right_dangerous:
                generated_direction = 0
            elif not is_front_dangerous and not is_right_dangerous:
                generated_direction = 1

    elif generated_direction == 0:
        if is_front_dangerous:
            if is_left_dangerous and not is_right_dangerous:
                generated_direction = 1
            elif not is_left_dangerous and is_right_dangerous:
                generated_direction = -1
            elif not is_left_dangerous and not is_right_dangerous:
                generated_direction = 1

    elif generated_direction == 1:
        if is_right_dangerous:
            if is_left_dangerous and not is_front_dangerous:
                generated_direction = 0
            elif not is_left_dangerous and is_front_dangerous:
                generated_direction = -1
            elif not is_left_dangerous and not is_front_dangerous:
                generated_direction = -1

    return generated_direction


# Just for testing
#input_shape = (30, 30)
#training_data_x, training_data_y = generate_examples_data_for_replay_memory(input_shape[0], input_shape[1], 1, 1, 0, 0)

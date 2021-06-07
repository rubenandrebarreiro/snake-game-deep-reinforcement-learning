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

# Import the Operative System Library as operative_system
import os as operative_system

from game.utils.plotting_helper import plot_board

operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the Tensorflow as Tensorflow alias Python's Module
import tensorflow as tensorflow

# Import the Backend Module from the TensorFlow.Python.Keras Python's Module
from tensorflow.python.keras import backend as keras_backend

# Import the boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs)
# from the Parameters and Arguments Python's Custom Module
from game.others.parameters_arguments import TENSORFLOW_KERAS_HPC_BACKEND_SESSION

# Import the Number of CPU's Processors/Cores
# from the Parameters and Arguments Python's Custom Module
from game.others.parameters_arguments import NUM_CPU_PROCESSORS_CORES

# Import the Number of GPU's Devices
# from the Parameters and Arguments Python's Custom Module
from game.others.parameters_arguments import NUM_GPU_DEVICES


# From the Game.Snake_Game Module,
# import the Snake Game object
import numpy as np
from keras import Sequential
from keras.layers import Dense

from game.snake_game import SnakeGame

# From the Game.Utils.Board_State_Utils Module,
# import the function to get the Distance to an Apple
from game.utils.board_state_utils import get_distance_from_snake_to_apple, print_board_status

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

    num_training_games = 1
    num_steps_per_game = 80

    snake_game = SnakeGame(width=width, height=height, border=1)

    for current_num_training_game in range(num_training_games):

        print("Starting the Generation of the Data for the Training Game #{}...\n"
              .format((current_num_training_game + 1)))

        board_state, reward, done, score_dictionary = snake_game.reset()

        # Retrieve the name of the Action taken
        action_name = {-1: "Turn Left", 0: "Straight Ahead", 1: "Turn Right"}

        # Plot the Board of the Snake Game for the initial state
        plot_board("{}.png".format(0), board_state, "Start")

        for current_num_step in range(num_steps_per_game):

            print("Generating the Data for the Step #{}..."
                  .format((current_num_step + 1)))

            snake_head_position = snake_game.snake[0]
            snake_body_positions = snake_game.snake[1:]
            apples_positions = snake_game.apples

            angle_with_apple, snake_direction_vector, \
                apple_direction_vector_normalized, snake_direction_vector_normalized = \
                get_angle_with_apple(snake_head_position, snake_body_positions, apples_positions[0])

            generated_direction, button_direction = \
                generate_direction_to_apple(snake_head_position, snake_body_positions, angle_with_apple)

            current_direction_vector, is_front_dangerous, is_left_dangerous, is_right_dangerous = \
                get_dangerous_directions(snake_head_position, snake_body_positions, board_state, snake_game.border)

            print("Front: ", is_front_dangerous)
            print("Left: ", is_left_dangerous)
            print("Right: ", is_right_dangerous)

            direction, button_direction, training_data_ys = \
                generate_training_data_ys(snake_head_position, snake_body_positions,
                                          button_direction, generated_direction, training_data_ys,
                                          is_front_dangerous, is_left_dangerous, is_right_dangerous)

            training_data_xs.append([is_left_dangerous, is_front_dangerous, is_right_dangerous,
                                     apple_direction_vector_normalized[0], snake_direction_vector_normalized[0],
                                     apple_direction_vector_normalized[1], snake_direction_vector_normalized[1]])

            board_state, reward, done, score_dictionary = snake_game.play_step(direction)

            if done:

                # Retrieve the current Score from the Dictionary of the Score
                current_score = score_dictionary["Score"]
                print("Game Over!!! Final Score: {}".format(current_score))
                print("\n")

                break

            # Plot the Board of the Game for the Action taken
            plot_board("{}.png".format(current_num_step+1), board_state, action_name[direction])

            # Retrieve the current Score from the Dictionary of the Score
            current_score = score_dictionary["Score"]

            print("Current Score: {}".format(current_score))
            print("\n")

    return training_data_xs, training_data_ys


def generate_training_data_ys(snake_head_position, snake_body_positions,
                              button_direction, generated_direction, training_data_ys,
                              is_front_dangerous, is_left_dangerous, is_right_dangerous):

    print(generated_direction)

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

    return generated_direction, button_direction, training_data_ys


# If the boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs) is set to True
if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:

    # Print the information about if the Model will be executed,
    # using High-Performance Computing (with CPUs and GPUs)
    print('\n')
    print('It will be used High-Performance Computing (with CPUs and GPUs):')
    print(' - Num. CPUS: ', NUM_CPU_PROCESSORS_CORES)
    print(' - Num. GPUS: ', NUM_GPU_DEVICES)
    print('\n')

    # Set the Configuration's Proto, for the given number of Devices (CPUs and GPUs)
    configuration_proto = \
        tensorflow.compat.v1.ConfigProto(device_count={'CPU': NUM_CPU_PROCESSORS_CORES,
                                                       'GPU': NUM_GPU_DEVICES})

    # Configure a TensorFlow Session for High-Performance Computing (with CPUs and GPUs)
    session = tensorflow.compat.v1.Session(config=configuration_proto)

    # Set the current Keras' Backend, with previously configured
    # TensorFlow Session for High-Performance Computing (with CPUs and GPUs)
    keras_backend.set_session(session)

training_data_x, training_data_y = generate_training_data(30, 30, 1, 1, 0, 0)

model = Sequential()
model.add(Dense(units=9, input_dim=7))

model.add(Dense(units=15, activation='relu'))
model.add(Dense(output_dim=3,  activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit((np.array(training_data_x).reshape(-1, 7)),
          (np.array(training_data_y).reshape(-1, 3)), batch_size=256, epochs=3)

model.save_weights('model.h5')
model_json = model.to_json()

with open('model.json', 'w') as json_file:
    json_file.write(model_json)

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

Snake Agent Module for the the Project

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
import time

operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# From the DateTime Library, import the DateTime Module
from datetime import datetime

# From the Collections Library,
# import the Deque
from collections import deque

# From the Game.Snake_Game Module,
# import the Snake Game
from game.snake_game import SnakeGame

# From the Game.Agent.Common.Snake_Trainer Module,
# import the Snake Game Q-Learning Trainer
from game.agent.common.snake_trainer import SnakeAgentQLearningTrainer

# From the Game.Agent.Common.Snake_Model Module,
# import the Snake Agent CNN (Convolutional Neural Network) Model
from game.agent.common.snake_model import SnakeAgentCNNModel

# From the Game.Utils.Plotting_Helper,
# import the Turn On functionality for the Interactive Mode
from game.utils.plotting_helper import turn_on_interactive_mode

# From the Game.Utils.Plotting_Helper,
# import the Turn Off functionality for the Interactive Mode
from game.utils.plotting_helper import turn_off_interactive_mode

# From the Game.Utils.Plotting_Helper,
# import the Plot Board
from game.utils.plotting_helper import plot_board

# From the Game.Utils.Plotting_Helper,
# import the Dynamic Training Plot
from game.utils.plotting_helper import dynamic_training_plot

# From the NumPy Library, import the Argmax function
from numpy import argmax

# From the NumPy Library, import the Random function
from numpy import random

# From the NumPy Library, import the Expand dim function
from numpy import expand_dims

# From the NumPy Library, import the Exponential function
from numpy import exp

# From the Random Library, import the Sample function
from random import sample

# From the Game.Others.Parameters_and_Arguments,
# import the Sleep Time, in seconds for
# a better presentation
from game.others.parameters_arguments import SLEEP_TIME_SECS

# From the Game.Others.Parameters_and_Arguments,
# import the Initial Learning Rates for
# the CNN (Convolutional Neural Network) Model
from game.others.parameters_arguments import INITIAL_LEARNING_RATES

# From the Game.Others.Parameters_and_Arguments,
# import the List of Available Optimisers to be used in
# the CNN (Convolutional Neural Network) Model
from game.others.parameters_arguments import AVAILABLE_OPTIMISERS_LIST

# From the Game.Others.Parameters_and_Arguments,
# import the Size of the Batch to be used for the Training of
# the CNN (Convolutional Neural Network) Model
from game.others.parameters_arguments import BATCH_SIZE

# From the Game.Others.Parameters_and_Arguments,
# import the Maximum Capacity for the Memory (Deque) of the Snake Agent
from game.others.parameters_arguments import MAX_MEMORY

# From the Game.Others.Parameters_and_Arguments,
# import the initial value for the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
from game.others.parameters_arguments import INITIAL_EPSILON_RANDOMNESS

# From the Game.Others.Parameters_and_Arguments,
# import the maximum value for the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
from game.others.parameters_arguments import MAXIMUM_EPSILON_RANDOMNESS

# From the Game.Others.Parameters_and_Arguments,
# import the Minimum value for the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
from game.others.parameters_arguments import MINIMUM_EPSILON_RANDOMNESS

# From the Game.Others.Parameters_and_Arguments,
# import the Decay Factor to adjust the value for
# the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
from game.others.parameters_arguments import DECAY_FACTOR_EPSILON_RANDOMNESS

# From the Game.Others.Parameters_and_Arguments,
# import the Gamma value (i.e., the Discount Reward)
from game.others.parameters_arguments import GAMMA_DISCOUNT_FACTOR

# From the Game.Others.Parameters_and_Arguments,
# import the Number of Games (Training Episodes)
from game.others.parameters_arguments import NUM_GAME_TRAINING_EPISODES


# Class for the Snake Agent
class SnakeAgent:

    # Constructor for the Snake Agent
    def __init__(self, optimiser_id, board_shape):

        self.last_action = 0

        self.num_consecutive_same_actions = 1

        # Initialise the Number of Games (Training Episodes) played by the Snake Agent
        self.num_games_episodes_played = 0

        # Initialise the Epsilon variable for the Randomness used to,
        # decide about Exploration and Exploitation
        self.epsilon_randomness = INITIAL_EPSILON_RANDOMNESS

        # Set the Gamma value (i.e., the Discount Reward)
        self.gamma_discount_factor = GAMMA_DISCOUNT_FACTOR

        # Set the Memory of the Snake Agent,
        # for the Deque structure with the Maximum Capacity defined for it
        self.memory = deque(maxlen=MAX_MEMORY)

        # Create the Snake Agent's Q-Learning Trainer, given the CNN (Convolutional Neural Network) Models,
        # for the current and target observations
        self.snake_q_learning_trainer = \
            SnakeAgentQLearningTrainer(learning_rate=INITIAL_LEARNING_RATES[optimiser_id],
                                       gamma_discount_factor=self.gamma_discount_factor)

        # Initialise the CNN (Convolutional Neural Network) Model for the Snake Agent,
        # for the current observations TODO - Confirm Input Shape and others
        self.snake_cnn_model_for_current_observations = \
            SnakeAgentCNNModel(AVAILABLE_OPTIMISERS_LIST[optimiser_id].lower(),
                               self.snake_q_learning_trainer.optimizer, board_shape, [16, 32], 3)

        # Initialise the CNN (Convolutional Neural Network) Model for the Snake Agent,
        # for the target observations TODO - Confirm Input Shape and others
        self.snake_cnn_model_for_target_observations = \
            SnakeAgentCNNModel(AVAILABLE_OPTIMISERS_LIST[optimiser_id].lower(),
                               self.snake_q_learning_trainer.optimizer, board_shape, [16, 32], 3)

        # Initialise the CNN (Convolutional Neural Network) Models for the Snake Agent,
        # for the current and target observations
        self.snake_q_learning_trainer.initialise_cnn_models(self.snake_cnn_model_for_current_observations,
                                                            self.snake_cnn_model_for_target_observations)

    # Function to remember a given tuple of the state of the Snake Agent,
    # by saving it in its Memory
    def remember(self, observation, action, reward, new_observation, done):

        # Append the given tuple of the state of the Snake Agent to its Memory
        self.memory.append([observation, action, reward, new_observation, done])

    # Function to train the Long (Replay/Experiences) Memory of the Snake Agent
    def train_long_replay_experiences_memory(self):

        # If the Memory of the Snake Agent has
        # the sufficient number of examples for sampling a Batch
        if len(self.memory) >= BATCH_SIZE:

            # Sample a random Batch of examples from the Memory of the Snake Agent
            mini_batch_sample = sample(self.memory, BATCH_SIZE)

        # If the Memory of the Snake Agent does not have
        # the sufficient number of examples for sampling a Batch
        else:

            # Set all the examples remembered by the Snake Agent
            mini_batch_sample = self.memory

        # Retrieve the tuple of states of the Snake Agent,
        # regarding the examples for sampling a Batch
        observations, actions, rewards, new_observations, dones = zip(*mini_batch_sample)

        # Train the Snake Agent, by a step performed by it
        self.snake_q_learning_trainer.train_step(observations, actions, rewards, new_observations, dones)

    # Function to train the Short Memory of the Snake Agent
    def train_short_memory(self, observation, action, reward, new_observation, done):

        # Train a Step for the Snake Agent's Q-Learning Trainer
        self.snake_q_learning_trainer.train_step(observation, action, reward, new_observation, done)

    # Function to make the Snake Agent take the next Action,
    # according to an Observation made by it
    def make_next_action(self, observation):

        # The Snake Agent decides to Explore the Environment
        if random.rand() <= self.epsilon_randomness:

            # Print some debug information
            print("The Snake Agent will explore the environment:\n")

            # Compute a random move, to the next Action to
            # be taken by the Snake Agent
            next_action = random.randint(-1, 2)

        # The Snake Agent decides to Exploit the Environment
        else:

            # Print some debug information
            print("The Snake Agent will exploit the environment:\n")

            observation_wrapper = expand_dims(observation, axis=0)

            # Predict the possible next Q-Values (Rewards)
            predicted_q_values = \
                self.snake_cnn_model_for_current_observations\
                    .model.predict(observation_wrapper).flatten()

            # Save the move that maximizes the Q-Values (Rewards),
            # for the next Action to be taken
            next_action = (argmax(predicted_q_values) - 1)

        if next_action == self.last_action:
            self.num_consecutive_same_actions += 1
        else:
            self.num_consecutive_same_actions = 1

        if self.num_consecutive_same_actions == 3:

            while self.last_action == next_action:

                # Compute a random move, to the next Action to
                # be taken by the Snake Agent
                next_action = random.randint(-1, 2)

        self.last_action = next_action

        # The Snake Agent decided to turn to left
        if next_action == -1:
            print("- The Snake Agent will turn left!!!\n")

        # The Snake Agent decided to go straight ahead
        elif next_action == 0:
            print("- The Snake Agent will go straight ahead!!!\n")

        # The Snake Agent decided to turn right
        elif next_action == 1:
            print("- The Snake Agent will turn right!!!\n")

        # Return the next Action for the Snake Agent
        return next_action


# The static function to train the Snake Agent
def train_snake_agent():

    # Create the Snake Game, for a Board Game of (30x30)
    snake_game = SnakeGame(30, 30, border=1)

    # Initialise the list of current Scores made by the Snake Agent
    current_scores = []

    # Initialise the list of Means of the current Scores made by the Snake Agent
    current_mean_scores = []

    # Initialise the current Total Score made by the Snake Agent
    current_total_score = 0

    # Initialise the current Score Record made by the Snake Agent
    current_score_record = 0

    # Initialise the Snake Agent
    snake_agent = SnakeAgent(2, snake_game.board_state().shape)  # TODO - 2 is the ID for the Adam Optimiser

    # Print a blank line
    print("\n")

    # Retrieve the name of the Actions possible to be taken by the Snake Agent
    action_names = {-1: "Turn Left", 0: "Straight Ahead", 1: "Turn Right"}

    # Retrieve the Now Datetime
    datetime_now = datetime.now()

    # Built the Timestamp for the Game
    game_start_timestamp = "{}_{}_{}_{}_{}_{}" \
        .format(datetime_now.year, datetime_now.month, datetime_now.day,
                datetime_now.hour, datetime_now.minute, datetime_now.second)

    # For each current Game (Training Episode)
    for current_num_game_episode in range(NUM_GAME_TRAINING_EPISODES):

        # Print some debug information
        print("------------------------ Game #{} ------------------------".format((current_num_game_episode + 1)))
        print("\n")
        print("Starting the current Game...\n")

        # Reset the Snake Game
        board_state, reward, done, info = snake_game.reset()

        # Initialise the Step counter, for the Snake Agent
        snake_agent_num_steps = 0

        # Built the Game Board path, to save the images of the Board of the Game
        game_board_path = "game_{}".format((current_num_game_episode + 1))

        # Build the Full Path of the Game Board
        game_board_full_path = "images/boards/{}/{}".format(game_start_timestamp, game_board_path)

        # Verify if the directory for the Full Path of the Game Board exists
        if not operative_system.path.exists(game_board_full_path):

            # Make the directory for the Full Path of the Game Board
            operative_system.makedirs(game_board_full_path)

        # Turn off the Interactive Mode of Plotting
        turn_off_interactive_mode()

        # Plot the Board of the Snake Game for the initial State
        plot_board("{}/step_0.png".format(game_board_full_path), board_state, "Start")

        # Turn on the Interactive Mode of Plotting
        turn_on_interactive_mode()

        # Print the initial State of the Snake Game
        snake_game.print_state()

        # Sleep of n second
        # Note: Uncomment/Comment if you want;
        time.sleep(SLEEP_TIME_SECS)

        # Print a blank line
        print("\n")

        # Start an infinite loop
        while True:

            # Print some debug information
            print("\n---------- Step #{} ----------\n".format((snake_agent_num_steps + 1)))

            # Retrieve the old observation made by the Snake Agent
            snake_old_observation = snake_game.board_state()

            # Make the Snake Agent take the next Action
            snake_action = snake_agent.make_next_action(snake_old_observation)

            # Make the Snake Agent take a step according to the next Action retrieved
            board_state, reward, done, score = snake_game.play_step(snake_action)

            # Increment the Step counter, for the Snake Agent
            snake_agent_num_steps += 1

            # Turn off the Interactive Mode of Plotting
            turn_off_interactive_mode()

            # Plot the Board of the Snake Game for the current State
            plot_board("{}/step_{}.png".format(game_board_full_path, snake_agent_num_steps),
                       board_state, action_names[snake_action])

            # Turn on the Interactive Mode of Plotting
            turn_on_interactive_mode()

            # Print the current State of the Snake Game
            snake_game.print_state()

            # Sleep of n second
            # Note: Uncomment/Comment if you want;
            time.sleep(SLEEP_TIME_SECS)

            # Print a blank line
            print("\n")

            # Retrieve the new observation made by the Snake Agent,
            # considering the next Action retrieved
            snake_new_observation = board_state

            # Remember the tuple for the last state of the Snake Agent
            snake_agent.remember(snake_old_observation, snake_action, reward,
                                 snake_new_observation, done)

            # Make the Snake Agent train its Short Memory, using the tuple for its last state
            snake_agent.train_short_memory(snake_old_observation, snake_action, reward,
                                           snake_new_observation, done)

            # If the current Game is over (i.e., Game Over situation)
            if done:

                # Print some debug information
                print("Game Over!!!\n")

                # Increase the number of Games played by the Snake Agent
                snake_agent.num_games_episodes_played += 1

                # Make the Snake Agent train its Long (Replay/Experiences) Memory
                snake_agent.train_long_replay_experiences_memory()

                # Retrieve the current Score from the Dictionary of the Score
                current_score = score["Score"]

                # If the current Score is greater than the Score's Record
                if current_score > current_score_record:

                    # Set the Score's Record as the current Score
                    current_score_record = current_score

                    # Save the Sequential Model for
                    # the CNN (Convolutional Neural Network) for the current observations
                    snake_agent.snake_cnn_model_for_current_observations.save_model()

                # Print the current Statistics for the Game,
                # regarding the last Action taken by the Snake Agent
                print("[ Game No.: {} | Score: {} | Record: {} ]"
                      .format(snake_agent.num_games_episodes_played, current_score, current_score_record))

                # Append the current Score to the list of the Scores
                current_scores.append(current_score)

                # Sum the current Score to the Total Score
                current_total_score += current_score

                # Compute the current Mean Score, taking into the account the Total Score made
                # and the number of Games played for the Snake Agent
                current_mean_score = (current_total_score / snake_agent.num_games_episodes_played)

                # Append the current Mean Score to the list of the Mean Scores
                current_mean_scores.append(current_mean_score)

                # Call the Dynamic Training Plot
                dynamic_training_plot(current_scores, current_mean_scores)

                # Break the loop, when the Training is done
                break

            # Print a separator
            print("-----------------------------\n")

        # Update the Epsilon variable for the Randomness used to,
        # decide about Exploration and Exploitation, taking into account both of
        # the Minimum and Maximum values for it, as also, to the Decay factor to adjust it
        """
        snake_agent.epsilon_randomness = \
            (MINIMUM_EPSILON_RANDOMNESS +
             ((MAXIMUM_EPSILON_RANDOMNESS - MINIMUM_EPSILON_RANDOMNESS) *
              exp(-DECAY_FACTOR_EPSILON_RANDOMNESS * current_num_game_episode)))
        """
        if snake_agent.epsilon_randomness > MINIMUM_EPSILON_RANDOMNESS:
            snake_agent.epsilon_randomness -= DECAY_FACTOR_EPSILON_RANDOMNESS


# The Main Function
if __name__ == '__main__':

    # Call the static method to train the Snake Agent,
    # using the Sequential Model for the CNN (Convolutional Neural Network)
    train_snake_agent()

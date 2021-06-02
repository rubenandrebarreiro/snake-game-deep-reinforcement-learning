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
# import the Dynamic Training Plot
from game.utils.plotting_helper import dynamic_training_plot

# From the NumPy Library, import the Argmax function
from numpy import argmax

# From the NumPy Library, import the Random function
from numpy import random

# From the Game.Others.Snake_Agent_Parameters,
# import the Maximum Capacity for the Memory (Deque) of the Snake Agent
from game.others.parameters_arguments import MAX_MEMORY

# From the Game.Others.Snake_Agent_Parameters,
# import the Initial Learning Rates for
# the CNN (Convolutional Neural Network) Model
from game.others.parameters_arguments import INITIAL_LEARNING_RATES

# From the Game.Others.Snake_Agent_Parameters,
# import the List of Available Optimisers to be used in
# the CNN (Convolutional Neural Network) Model
from game.others.parameters_arguments import AVAILABLE_OPTIMISERS_LIST

# From the Game.Others.Snake_Agent_Parameters,
# import the Size of the Batch to be used for the Training of
# the CNN (Convolutional Neural Network) Model
from game.others.parameters_arguments import BATCH_SIZE


# Class for the Snake Agent
class SnakeAgent:

    # Constructor for the Snake Agent
    def __init__(self, optimiser_id):

        # Initialise the number of Games played by the Snake Agent
        self.num_games = 0

        # Initialise the Epsilon variable for the Randomness used to,
        # decide about Exploration and Exploitation
        self.epsilon_randomness = 0

        # Set the Gamma value (i.e., the Discount Reward)
        self.gamma_discount_factor = 0.9

        # Set the Memory of the Snake Agent,
        # for the Deque structure with the Maximum Capacity defined for it
        self.memory = deque(maxlen=MAX_MEMORY)

        # Create the Q-Learning Trainer for the Snake Agent
        self.snake_q_learning_trainer = \
            SnakeAgentQLearningTrainer(self.snake_cnn_model, learning_rate=INITIAL_LEARNING_RATES[optimiser_id],
                                       gamma_discount_factor=self.gamma_discount_factor)

        # TODO - Confirmar Input
        # Initialise the CNN (Convolutional Neural Network) for the Snake Agent
        self.snake_cnn_model = \
            SnakeAgentCNNModel(AVAILABLE_OPTIMISERS_LIST[optimiser_id].lower(),
                               self.snake_q_learning_trainer.optimizer, 11, [16, 32], 3)

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
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        # If the Memory of the Snake Agent does not have
        # the sufficient number of examples for sampling a Batch
        else:

            # Set all the examples remembered by the Snake Agent
            mini_sample = self.memory

        # Retrieve the tuple of states of the Snake Agent,
        # regarding the examples for sampling a Batch
        observations, actions, rewards, new_observations, dones = zip(*mini_sample)

        # Train the Snake Agent, by a step performed by it
        self.snake_q_learning_trainer.train_step(observations, actions, rewards, new_observations, dones)

    # Function to train the Short Memory of the Snake Agent
    def train_short_memory(self, observation, action, reward, new_observation, done):
        self.snake_q_learning_trainer.train_step(observation, action, reward, new_observation, done)

    # Function to make the Snake Agent take the next Action,
    # according to an Observation made by it
    def make_next_action(self, observation):

        # Decrease the Epsilon variable for the Randomness used to,
        # decide about Exploration and Exploitation,
        # according to the current number of Games played by the Snake Agent
        self.epsilon_randomness = 80 - self.num_games

        # Initialise the vector for the next Action to
        # be taken by the Snake Agent
        next_action = [0, 0, 0]

        # The Snake Agent decides to Explore the Environment
        if random.rand() <= self.epsilon_randomness:

            # Compute a random move, to the next Action to
            # be taken by the Snake Agent
            move = random.randint(0, 2)

        # The Snake Agent decides to Exploit the Environment
        else:

            # Create a Tensor for the Observations
            observation_tensor = observation.reshape([1, observation.shape[0]])

            # Predict the possible next Q-Values (Rewards)
            predicted = self.snake_cnn_model.model.predict(observation_tensor).flatten()

            # Save the action that maximizes the Q-Value (Reward)
            move = argmax(predicted)

        # Set the next Action to be taken by the Snake Agent,
        # according to the move chosen by it
        next_action[move] = 1

        # Return the next Action for the Snake Agent
        return next_action


# The static function to train the Snake Agent
def train_snake_agent():

    # Initialise the list of current Scores made by the Snake Agent
    current_scores = []

    # Initialise the list of Means of the current Scores made by the Snake Agent
    current_mean_scores = []

    # Initialise the current Total Score made by the Snake Agent
    current_total_score = 0

    # Initialise the current Score Record made by the Snake Agent
    current_score_record = 0

    # Initialise the Snake Agent
    snake_agent = SnakeAgent(3)  # TODO - 3 is the ID for the Adam Optimiser

    # Create the Snake Game, for a Board Game of (30x30)
    snake_game = SnakeGame(30, 30, border=1)

    # Start an infinite loop
    while True:

        # Retrieve the old observation made by the Snake Agent
        snake_old_observation = snake_game.get_state()

        # Make the Snake Agent take the next Action
        snake_action = snake_agent.make_next_action(snake_old_observation)

        # Make the Snake Agent take a step according to the next Action retrieved
        board_state, reward, done, score = snake_game.play_step(snake_action)

        # Retrieve the new observation made by the Snake Agent,
        # considering the next Action retrieved
        snake_new_observation = snake_game.get_state()

        # Make the Snake Agent train its Short Memory, using the tuple for its last state
        snake_agent.train_short_memory(snake_old_observation, snake_action, reward, snake_new_observation, done)

        # Remember the tuple for the last state of the Snake Agent
        snake_agent.remember(snake_old_observation, snake_action, reward, snake_new_observation, done)

        # If the current Game is over (i.e., Game Over situation)
        if done:

            # train long memory (replay memory or experience replay)
            snake_game.reset()

            # Increase the number of Games played by the Snake Agent
            snake_agent.num_games += 1

            # Make the Snake Agent train its Long (Replay/Experiences) Memory
            snake_agent.train_long_replay_experiences_memory()

            # If the current Score is greater than the Score's Record
            if score > current_score_record:

                # Set the Score's Record as the current Score
                current_score_record = score

                # Save the Sequential Model for
                # the CNN (Convolutional Neural Network)
                snake_agent.snake_cnn_model.save_model()

            # Print the current Statistics for the Game,
            # regarding the last Action taken by the Snake Agent
            print("[ Game No.: {} | Score: {} | Record: {} ]"
                  .format(snake_agent.num_games, score, current_score_record))

            # Append the current Score to the list of the Scores
            current_scores.append(score)

            # Sum the current Score to the Total Score
            current_total_score += score

            # Compute the current Mean Score, taking into the account the Total Score made
            # and the number of Games played for the Snake Agent
            current_mean_score = (current_total_score / snake_agent.num_games)

            # Append the current Mean Score to the list of the Mean Scores
            current_mean_scores.append(current_mean_score)

            # Call the Dynamic Training Plot
            dynamic_training_plot(current_scores, current_mean_scores)


# The Main Function
if __name__ == '__main__':

    # Call the static method to train the Snake Agent,
    # using the Sequential Model for the CNN (Convolutional Neural Network)
    train_snake_agent()

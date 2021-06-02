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

# From the Collections Library, import the Deque Module
from collections import deque

from game.snake_game import SnakeGame

from snake_trainer import SnakeAgentQLearningTrainer

from snake_model import SnakeAgentCNNModel

from game.utils.plotting_helper import dynamic_training_plot

from numpy import argmax, random

from cnn_parameters import OPTIMISERS_LIST

MAX_MEMORY = 100000

BATCH_SIZE = 1000
LEARNING_RATE = 0.001


# Class for the Snake Agent
class SnakeAgent:

    # Constructor for the Snake Agent
    def __init__(self):

        # Initialise the number of Games played by the Snake Agent
        self.num_games = 0

        # Initialise the Epsilon variable for the Randomness used to,
        # decide about Exploration and Exploitation
        self.epsilon_randomness = 0

        # Set the Gamma value (i.e., the Discount Reward)
        self.gamma_discount_factor = 0.9

        # Set the
        self.memory = deque(maxlen=MAX_MEMORY)

        self.snake_q_learning_trainer = \
            SnakeAgentQLearningTrainer(self.snake_cnn_model, learning_rate=LEARNING_RATE,
                                       gamma_discount_factor=self.gamma_discount_factor)

        # TODO - Confirmar Input
        self.snake_cnn_model = \
            SnakeAgentCNNModel(OPTIMISERS_LIST[0].lower(), self.snake_q_learning_trainer.optimizer, 11, [16, 32], 3)

    def remember(self, observation, action, reward, new_observation, done):
        self.memory.append([observation, action, reward, new_observation, done])

    def train_long_replay_experiences_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        observations, actions, rewards, new_observations, dones = zip(*mini_sample)
        self.snake_q_learning_trainer.train_step(observations, actions, rewards, new_observations, dones)

    def train_short_memory(self, observation, action, reward, new_observation, done):
        self.snake_q_learning_trainer.train_step(observation, action, reward, new_observation, done)

    def get_next_action(self, observation):

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
    scores = []

    # Initialise the list of Means of the current Scores made by the Snake Agent
    mean_scores = []

    # Initialise the current Total Score made by the Snake Agent
    total_score = 0

    # Initialise the current Score Record made by the Snake Agent
    score_record = 0

    # Initialise the Snake Agent
    snake_agent = SnakeAgent()

    # Create the Snake Game, for a Board Game of (30x30)
    snake_game = SnakeGame(30, 30, border=1)

    # Start an infinite loop
    while True:

        # Retrieve the old observation made by the Snake Agent
        snake_old_observation = snake_game.get_state()

        # Get the next Action for the Snake Agent
        snake_action = snake_agent.get_next_action(snake_old_observation)

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
            if score > score_record:

                # Set the Score's Record as the current Score
                score_record = score

                # Save the Sequential Model for
                # the CNN (Convolutional Neural Network)
                snake_agent.snake_cnn_model.save_model()

            # Print the current Statistics for the Game,
            # regarding the last Action taken by the Snake Agent
            print("[ Game No.: {} | Score: {} | Record: {} ]".format(snake_agent.num_games, score, score_record))

            # Append the current Score to the list of the Scores
            scores.append(score)

            # Sum the current Score to the Total Score
            total_score += score

            # Compute the current Mean Score, taking into the account the Total Score made
            # and the number of Games played for the Snake Agent
            mean_score = (total_score / snake_agent.num_games)

            # Append the current Mean Score to the list of the Mean Scores
            mean_scores.append(mean_score)

            # Call the Dynamic Training Plot
            dynamic_training_plot(scores, mean_scores)


# The Main Function
if __name__ == '__main__':

    # Call the static method to train the Snake Agent,
    # using the Sequential Model for the CNN (Convolutional Neural Network)
    train_snake_agent()

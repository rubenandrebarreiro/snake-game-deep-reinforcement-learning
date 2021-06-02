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

from parameters import OPTIMISERS

MAX_MEMORY = 100000

BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class SnakeAgent:

    def __init__(self):

        self.num_games = 0
        self.epsilon_randomness = 0
        self.gamma_discount_factor = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.snake_q_learning_trainer = \
            SnakeAgentQLearningTrainer(self.snake_cnn_model, learning_rate=LEARNING_RATE,
                                       gamma_discount_factor=self.gamma_discount_factor)

        # TODO - Confirmar
        self.snake_cnn_model = \
            SnakeAgentCNNModel(OPTIMISERS[0].lower(), self.snake_q_learning_trainer.optimizer, 11, [16, 32], 3)

    def remember(self, observation, action, reward, new_observation, done):
        self.memory.append([observation, action, reward, new_observation, done])

    def train_long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        observations, actions, rewards, new_observations, dones = zip(*mini_sample)
        self.snake_q_learning_trainer.train_step(observations, actions, rewards, new_observations, dones)

    def train_short_memory(self, observation, action, reward, new_observation, done):
        self.snake_q_learning_trainer.train_step(observation, action, reward, new_observation, done)

    def get_next_action(self, observation):
        # random moves: tradeoff exploration / exploitation
        self.epsilon_randomness = 80 - self.num_games
        next_action = [0, 0, 0]

        # Explore
        if random.rand() <= self.epsilon_randomness:
            move = random.randint(0, 2)

        # Exploit
        else:

            # Create a Tensor for the Observations
            observation_tensor = observation.reshape([1, observation.shape[0]])

            # Predict the possible next Q-Values (Rewards)
            predicted = self.snake_cnn_model.model.predict(observation_tensor).flatten()

            # Save the action that maximizes the Q-Value (Reward)
            move = argmax(predicted)

        next_action[move] = 1

        return next_action


def train_snake_agent():

    scores = []
    mean_scores = []
    total_score = 0
    score_record = 0
    snake_agent = SnakeAgent()

    # Create the Snake Game, for a Board Game of (30x30)
    snake_game = SnakeGame(30, 30, border=1)

    while True:

        snake_old_observation = snake_game.get_state()

        snake_action = snake_agent.get_next_action(snake_old_observation)

        board_state, reward, done, score = snake_game.play_step(snake_action)

        snake_new_observation = snake_game.get_state()

        snake_agent.train_short_memory(snake_old_observation, snake_action, reward, snake_new_observation, done)

        snake_agent.remember(snake_old_observation, snake_action, reward, snake_new_observation, done)

        if done:

            # train long memory (replay memory or experience replay)
            snake_game.reset()

            snake_agent.num_games += 1
            snake_agent.train_long_memory()

            if score > score_record:
                score_record = score
                snake_agent.snake_cnn_model.save_model()

            print("[ Game No.: {} | Score: {} | Record: {} ]".format(snake_agent.num_games, score, score_record))

            scores.append(score)
            total_score += score
            mean_score = (total_score / snake_agent.num_games)
            mean_scores.append(mean_score)
            dynamic_training_plot(scores, mean_scores)


if __name__ == '__main__':
    train_snake_agent()

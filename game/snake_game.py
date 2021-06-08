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

# Import NumPy as NumPy
import numpy as numpy

# Import the Random Integer Generator Module,
# from the NumPy Library, with the random_int alias
from numpy.random import randint as random_int


# Class for the Snake Game
class SnakeGame:

    # Constructor of the Snake Game
    def __init__(self, width, height, food_amount=1,
                 border=0, grass_growth=0, max_grass=0):

        # Initialize the Board of the Game
        self.width = width
        self.height = height
        self.board = (numpy.zeros((height, width, 3), dtype=numpy.float32))
        self.food_amount = food_amount
        self.border = border
        self.grass_growth = grass_growth
        self.grass = (numpy.zeros((height, width)) + max_grass)
        self.max_grass = max_grass
        self.direction = -1
        self.snake = []
        self.apples = []
        self.score = 0
        self.done = False
        self.reset()

    # Create the Apples on the Board of the Game
    def create_apples(self):

        # Create a new Apple away from the Snake
        while len(self.apples) < self.food_amount:

            # Generate a random coordinate (x, y) for the current Apple
            apple = (random_int(0, (self.height - 1)), random_int(0, (self.width - 1)))

            # While the generated random coordinate (x, y) for the current Apple
            # is a position already filled by the Snake
            while apple in self.snake:

                # Generate another random coordinate (x, y) for the current Apple
                apple = (random_int(0, (self.height - 1)), random_int(0, (self.width - 1)))

            # Append the current generated Apple to the list of them
            self.apples.append(apple)

    # Function to create the Snake
    def create_snake(self):

        # Create a snake, size 5, at random position and orientation
        # NOTE: Use a threshold for the generated position,
        #       in order to ensure that the Snake is not too close to the Border of the Game;
        position_x = random_int(5, (self.width - 5))
        position_y = random_int(5, (self.height - 5))

        # Generate a random integer number, to decide the initial direction of the Snake
        self.direction = random_int(0, 4)

        # The list of the (x,y) coordinates filled by the Snake
        self.snake = []

        # For each Position of the Snakes' Body
        for _ in range(5):

            # If the Snake is directed to up
            if self.direction == 0:
                position_y = (position_y + 1)

            # If the Snake directed to left
            elif self.direction == 1:
                position_x = (position_x - 1)

            # If the Snake directed to down
            elif self.direction == 2:
                position_y = (position_y - 1)

            # If the Snake directed to right
            elif self.direction == 3:
                position_x = (position_x + 1)

            # Append the current (x,y) coordinate of the body of
            # the Snake in the Board of the Game
            self.snake.append((position_y, position_x))

    # Function to grow the Snake, after it eats an Apple
    def grow_snake(self, direction):

        # Add one position to Snake's head,
        # according to its direction:
        # -> 0 = up
        # -> 1 = right
        # -> 2 = down
        # -> 3 = left
        position_y, position_x = self.snake[0]

        # The Snake go up
        if direction == 0:
            position_y = (position_y - 1)

        # The Snake go right
        elif direction == 1:
            position_x = (position_x + 1)

        # The Snake go down
        elif direction == 2:
            position_y = (position_y + 1)

        # The Snake go left
        else:
            position_x = (position_x - 1)

        # Insert a new (x, y) coordinate to the body of the Snake
        self.snake.insert(0, (position_y, position_x))

    # Function to check if the Snake collided or not
    def check_collisions(self):

        # Retrieve the (x, y) coordinates of the Snake
        x, y = self.snake[0]

        # Check if game is over by colliding with edge or itself
        # just need to check snake's head
        if ((x == -1) or (x == self.height)
                or (y == -1) or (y == self.width)
                or (x, y) in self.snake[1:]):

            # Set the Game as done
            self.done = True

    # Function for the Snake take an action
    def play_step(self, action):

        # Move snake/game by one step.
        # The resulting action can be:
        # -> -1 (turn left);
        # -> 0 (continue);
        # -> 1 (turn right)
        direction = int(action)

        # Assert if the random number generated for the direction of the Snake,
        # is inside the interval [-1,1]
        assert (-1 <= direction <= 1)

        # Sum the direction of the Snake
        self.direction += direction

        # If the direction of the Snake is negative
        if self.direction < 0:

            # Set the direction of the Snake as 3
            self.direction = 3

        # If the direction of the Snake is positive
        elif self.direction > 3:

            # Set the direction of the Snake as 0
            self.direction = 0

        # The Snake will grow through the defined direction
        # Two steps: grow + remove last
        self.grow_snake(self.direction)

        # If the Snake eaten one of the Apples
        if self.snake[0] in self.apples:

            # Remove the Snake's Head from the list of the Apples
            self.apples.remove(self.snake[0])

            # Set the reward for the Snake eaten an Apple, as 1
            reward = 1

            # Generate a new Apple
            self.create_apples()

        # If the Snake does not eaten one of the Apples
        else:

            # Pop the Snake's tail
            self.snake.pop()
            self.check_collisions()

            # If the Game is over, the reward will be -1
            if self.done:
                reward = -1

            # If the Game is not over, the reward will be 0
            else:
                reward = 0

        # If the Game is not over
        if reward >= 0:

            # Retrieve the Snake's head
            x, y = self.snake[0]

            # Sum to the reward, the (x,y) coordinates of the Grass
            reward += self.grass[x, y]

            # Set the Grass in the (x,y) coordinates of the Snake's Head, as 0
            self.grass[x, y] = 0

            # Sum the reward to the current Score
            self.score += reward

            # Make the Grass grow
            self.grass += self.grass_growth

            # Set the maximum for the growth of the Grass,
            # for the cases of the Grass cannot grow anymore
            self.grass[(self.grass > self.max_grass)] = self.max_grass

        # Return the State of the Board, the current Reward, the done Flag and current Score
        return self.board_state(), reward, self.done, {"Score": self.score}

    # Function to get the current state of the Game
    def get_state(self):

        # Easily get current state (score, apple, Snake's Head and Tail)
        score = self.score
        apple = self.apples
        head = self.snake[0]
        tail = self.snake[1:]

        # Return the current state of the Game
        return score, apple, head, tail, self.direction

    # Print the state the
    def print_state(self):

        # For each row of the Board
        for i in range(self.height):

            # Print a normal line
            line = ("." * self.width)

            # For the (x,y) coordinates of each Apple
            for x, y in self.apples:

                # If the y coordinate of the Apple is equal to the current row
                if y == i:
                    # Print an "A" in the place of the Apple
                    line = (line[:x] + "A" + line[(x + 1):])

            # For the (x,y) coordinates of the Snake
            for x, y in self.snake:

                # If the y coordinate of the Snake is equal to the current row
                if y == i:
                    # Print a "X" in the place of the Snake's body part
                    line = (line[:x] + "X" + line[(x + 1):])

            # Print the current row (or, line)
            print(line)

    # Function to test the Snake taking a step
    def test_step(self, direction):

        # To test:
        # - Move the Snake;
        # - Print the state of the Game;
        self.play_step(direction)
        self.print_state()

        # If the Game is done
        if self.done:

            # Print the information about the Game being over,
            # as also, the Final Score achieved
            print("Game Over! Score=", self.score)

    # Function to reset the state of the Game
    def reset(self):

        # Reset state of the Game
        self.score = 0
        self.done = False
        self.create_snake()
        self.apples = []
        self.create_apples()
        self.grass[:, :] = self.max_grass

        # Return the new state of the Game
        return self.board_state(), 0, self.done, {"Score": self.score}

    # Function to compute the current state of the Game, on its Board
    def board_state(self, mode="human", close=False):

        # Render the environment
        self.board[:, :, :] = 0

        # If there is some maximum threshold for the Grass growing
        if self.max_grass > 0:

            # Paint that (x,y) coordinate as Green,
            # according to the threshold for the Grass growing
            self.board[:, :, 1] = (self.grass / self.max_grass * 0.3)

        # If the Game is not done
        if not self.done:

            # Retrieve the Snake's Head
            x, y = self.snake[0]

            # Paint all the (x,y) coordinates as White
            self.board[x, y, :] = 1

        # For each (x,y) coordinate of the Snake's body
        for x, y in self.snake[1:]:

            # Paint the current (x,y) coordinate of the Snake's body as Red
            self.board[x, y, 0] = 1

        # For each (x,y) coordinate of the Apples
        for x, y in self.apples:

            # Paint the current (x,y) coordinate of the Apples as Green
            self.board[x, y, 1] = 1

        # If there is no Border of the Board of the Snake Game
        if self.border == 0:

            # Return the Board of the Snake Game
            return self.board

        # If there is no Border of the Board of the Snake Game
        else:

            # Retrieve the Height and the Width of the shape of the Board of the Snake Game
            height, width, _ = self.board.shape

            # Fulfill the Board of the Snake Game with the Border
            board = numpy.full(((height + (self.border * 2)), (width + (self.border * 2)), 3), 0.5, numpy.float32)

            # Update the Board of the Snake Game
            board[self.border:-self.border, self.border:-self.border] = self.board

            # Return the Board of the Snake Game
            return board


# Just run this if this file is the main
#if __name__ == '__main__':

    # Create the Board (m x n) for the Snake Game
#    game = SnakeGame(20, 20)

    # Print the information of the Board for the Snake Game,
    # in the current Step
#    game.print_state()

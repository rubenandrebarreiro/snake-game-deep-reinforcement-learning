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

Snake Game Demonstration
(given by the Instructor(s))

"""

# Import Python's Modules, Libraries and Packages

# Import PyPlot from the Matplotlib Python's Library
import matplotlib.pyplot as py_plot

# Import the Snake Game, with the Snake_Game alias
from game.snake_game import SnakeGame as Snake_Game


# Function to plot the Board of the Snake Game
def plot_board(file_name, board, text=None):

    # Initialise the Figure for Plotting
    py_plot.figure(figsize=(10, 10))

    # Show the Plotting for the Board of the Snake Game
    py_plot.imshow(board)

    # Deactivate the Plotting of the Axis
    py_plot.axis("off")

    # If the text is not None
    if text is not None:

        # Plot the Text for the Action taken
        py_plot.gca().text(3, 3, text, fontsize=45, color="yellow")

    # Save the Image for the Plotting of the Board of the Snake Game
    py_plot.savefig(file_name, bbox_inches="tight")

    # Close the Plotting for the Board of the Snake Game
    py_plot.close()


# Build a Demo for the Snake Game
def snake_demo(actions):

    # Create the Snake Game, for a Board Game of (30x30)
    snake_game = Snake_Game(30, 30, border=1)

    # Reset the Snake Game
    board, reward, done, info = snake_game.reset()

    # Retrieve the name of the Action taken
    action_name = {-1: "Turn Left", 0: "Straight Ahead", 1: "Turn Right"}

    # Plot the Board of the Snake Game for the initial state
    plot_board("images/0.png", board, "Start")

    # For each Frame and Action taken
    for frame, action in enumerate(actions):

        # Take the respective current Step, according to the Action taken
        board, reward, done, info = snake_game.step(action=action)

        # Plot the Board of the Game for the Action taken
        plot_board(f"images/{(frame + 1)}.png", board, action_name[action])


# Initialise the Demo for the Snake Game
snake_demo([0, 1, 0, -1])

"""

Deep Learning - Assignment #2:
- Snake Game - Deep Reinforcement Learning

Integrated Master of Computer Science and Engineering

NOVA School of Science and Technology,
NOVA University of Lisbon - 2020/2021

Authors:
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt) - Student no. 42648

Instructor(s):
- Ludwig Krippahl (ludi@fct.unl.pt)
- Claudia Soares (claudia.soares@fct.unl.pt)

Graphics Module for the the Project

"""

# Import Python's Modules, Libraries and Packages

# Import PyPlot from the Matplotlib Python's Library
from matplotlib import pyplot as py_plot


# Function to plot the Board of the Snake Game, for a given Step taken
def plot_board(game_board, current_score, current_step, done, full_screen=True):

    # If the Board is to be displayed in Full-Screen
    if full_screen:

        # Get the current Figure Manager
        figure_manager = py_plot.get_current_fig_manager()

        # Set the Window of the Figure Manager, as maximized
        figure_manager.window.showMaximized()

    # If the Game is not done yet
    if not done:

        # Plot the title for the Board of the Snake Game,
        # for the current given Step taken
        py_plot.title("Snake Game\nCurrent Step: {} ; Current Score: {}"
                      .format(current_step, current_score), color="green")

        # Plot the Board of the Snake Game
        py_plot.imshow(game_board)

        # Show the Plot of the Board of the Snake Game,
        # with no blocking
        py_plot.show(block=False)

        # Pause/Freeze the Plot of the Board of the Snake Game,
        # for 0.5 seconds
        py_plot.pause(0.5)

    # If the Game is already done
    else:

        # For 10 "dummy" steps
        for index in range(10):

            # If the current "dummy" step is even
            if (index % 2) == 0:

                # Plot the title for the Board of the Snake Game,
                # for the current given Step taken, giving the information of Game Over
                py_plot.title("Snake Game\nGame Over - Final Step: {} ; Final Score: {}"
                              .format(current_step, current_score), color="red")

            # If the current "dummy" step is odd
            else:

                # Plot the title for the Board of the Snake Game,
                # for the current given Step taken
                py_plot.title("Snake Game\n", color="green")

                # Plot the Board of the Snake Game
                py_plot.imshow(game_board)

                # Show the Plot of the Board of the Snake Game,
                # with no blocking
                py_plot.show(block=False)

                # Pause/Freeze the Plot of the Board of the Snake Game,
                # for 0.25 seconds
                py_plot.pause(0.25)

        # Close the Plot of the Board of the Snake Game
        py_plot.close()

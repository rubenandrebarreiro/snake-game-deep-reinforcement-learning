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

# From the IPython Library, import the Display Module
from IPython import display

# From the Matplotlib Library import the PyPlot Module, as the py_plot alias
import matplotlib.pyplot as py_plot

# Turn the Interactive Mode of the PyPlot Module
py_plot.ion()


# Function for the Dynamic Training Plot
def dynamic_training_plot(scores, mean_scores):

    # Clear the output of the Display
    display.clear_output(wait=True)

    # Display the current figure for the Plotting
    display.display(py_plot.gcf())

    # Clear the current figure of the Plot
    py_plot.clf()

    # Set the title of the Plot
    py_plot.title("Training the Convolutional Neural Network...")

    # Plot the Labels of the X-Axis and Y-Axis
    py_plot.xlabel("Number of Games")
    py_plot.ylabel("Scores/Mean Scores")

    # Plot the curves for the Scores
    py_plot.plot(scores, linestyle="-", color="cyan", label="Scores")
    py_plot.plot(mean_scores,  linestyle="--", color="orange", label="Mean Scores")

    # Plot the Legend of the curves for the Scores
    py_plot.legend()

    # Set the limit of the Y-Axis, as 0
    py_plot.ylim(ymin=0)

    # Plot the text tags for the current Scores and Mean Scores
    py_plot.text((len(scores) - 1), scores[-1], str(scores[-1]))
    py_plot.text((len(mean_scores) - 1), mean_scores[-1], str(mean_scores[-1]))

    # Show the plotting
    py_plot.show(block=False)

    # Pause the Plotting, for 0.1 seconds
    py_plot.pause(0.1)

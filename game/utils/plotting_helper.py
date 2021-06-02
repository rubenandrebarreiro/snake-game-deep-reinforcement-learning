from IPython import display

import matplotlib.pyplot as py_plot

py_plot.ion()


def dynamic_training_plot(scores, mean_scores):

    display.clear_output(wait=True)
    display.display(py_plot.gcf())

    py_plot.clf()

    py_plot.title("Training the Convolutional Neural Network...")

    py_plot.xlabel("Number of Games")
    py_plot.ylabel("Scores/Mean Scores")

    py_plot.plot(scores, linestyle="-", color="cyan", label="Scores")
    py_plot.plot(mean_scores,  linestyle="--", color="orange", label="Mean Scores")

    py_plot.legend()

    py_plot.ylim(ymin=0)

    py_plot.text((len(scores) - 1), scores[-1], str(scores[-1]))
    py_plot.text((len(mean_scores) - 1), mean_scores[-1], str(mean_scores[-1]))

    py_plot.show(block=False)

    py_plot.pause(0.1)

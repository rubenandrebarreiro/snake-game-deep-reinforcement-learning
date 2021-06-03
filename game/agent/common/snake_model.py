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
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# From the DateTime Library, import the DateTime Module
from datetime import datetime

# From the Game's CNN (Convolutional Neural Network) Parameters,
# import the list of the Kernel Sizes
from game.others.parameters_arguments import KERNEL_SIZES_LIST

# From the Game's CNN (Convolutional Neural Network) Parameters,
# import the list of the Strides
from game.others.parameters_arguments import STRIDES_LIST

# Import the TensorFlow Library, with tensorflow alias
import tensorflow as tensorflow

# From the TensorFlow.Keras Module,
# import the Sequential Learning Model
from tensorflow.keras import Sequential

# From the TensorFlow.Keras.Layers Module,
# import the Convolutional 2D Layer
from tensorflow.keras.layers import Conv2D

# From the TensorFlow.Keras.Layers Module,
# import the Maximum Pooling 2D Layer
from tensorflow.keras.layers import MaxPooling2D

# From the TensorFlow.Keras.Layers Module,
# import the Activation Layer
from tensorflow.keras.layers import Activation

# From the TensorFlow.Keras.Layers Module,
# import the Dense Layer
from tensorflow.keras.layers import Dense

# From the TensorFlow.Keras.Initializers Module,
# import the He Uniform Initializer
from tensorflow.keras.initializers import HeUniform


# Class for the Snake Agent's CNN (Convolutional Neural Network) Model
class SnakeAgentCNNModel:

    # Constructor of the Snake Agent's CNN (Convolutional Neural Network) Model
    def __init__(self, optimizer_for_model_name, optimizer_for_model,
                 input_shape, hidden_shapes, output_shape):

        # Initialise the Sequential Model as None
        self.model = None

        # Set the name of the Optimizer chosen
        self.optimizer_for_model_name = optimizer_for_model_name

        # Set the Optimizer chosen
        self.optimizer_for_model = optimizer_for_model

        # Set the Input Shape of the CNN (Convolutional Neural Network)
        self.input_shape = input_shape

        # Set the Hidden Shapes (Units) for the Convolutional Layers
        self.hidden_shapes = hidden_shapes

        # Set the Output Shape of the CNN (Convolutional Neural Network)
        self.output_shape = output_shape

    # Function to compute the Sequential Model,
    # for the mapping of the Observations/States to Actions
    def compute_model(self):

        # Initialise the He Uniform Initializer for the Model
        he_uniform_initializer = HeUniform()

        # Initialise the Sequential Model
        self.model = Sequential()

        # For each Hidden Layer of the CNN (Convolutional Neural Network)
        for hidden_shape_index in range(len(self.hidden_shapes)):

            # If it is the 1s Hidden Layer Shape
            if hidden_shape_index == 0:

                # Add a 2D Convolutional Layer, for the 1s Hidden Layer (with the Input Shape),
                # with a He Uniform Initializer
                self.model.add(Conv2D(self.hidden_shapes[hidden_shape_index],
                                      kernel_size=(KERNEL_SIZES_LIST[hidden_shape_index], KERNEL_SIZES_LIST[hidden_shape_index]),
                                      strides=(STRIDES_LIST[hidden_shape_index], STRIDES_LIST[hidden_shape_index]),
                                      kernel_initializer=he_uniform_initializer, padding="same",
                                      input_shape=self.input_shape))

            # If it is not the 1s Hidden Layer Shape
            else:

                # Add a 2D Convolutional Layer, for the remaining Hidden Layers,
                # with a He Uniform Initializer
                self.model.add(Conv2D(self.hidden_shapes[hidden_shape_index],
                                      kernel_size=(KERNEL_SIZES_LIST[hidden_shape_index], KERNEL_SIZES_LIST[hidden_shape_index]),
                                      strides=(STRIDES_LIST[hidden_shape_index], STRIDES_LIST[hidden_shape_index]),
                                      kernel_initializer=he_uniform_initializer, padding="same"))

            # Add a Maximum 2D Pooling Layer
            self.model.add(MaxPooling2D((2, 2)))

            # Add a ReLU Activation Layer
            self.model.add(Activation("relu"))

        # Add a Dense Layer, with 256 Units, with the He Uniform Initializer
        self.model.add(Dense(256, kernel_initializer=he_uniform_initializer))

        # Add a Dense Layer, for the Shape of the Actions Vector
        self.model.add(Dense(self.output_shape))

        # Compile the Sequential Model, with the Huber Loss, using the chosen Optimiser
        self.model.compile(loss=tensorflow.keras.losses.Huber(),
                           optimizer=self.optimizer_for_model,
                           metrics=["accuracy"])

    # Function to save the Sequential Model
    def save_model(self):

        # Retrieve the Now Datetime
        datetime_now = datetime.now()

        # Retrieve the Timestamp to save the Sequential Model
        model_timestamp = "{}_{}_{}_{}_{}_{}"\
            .format(datetime_now.year, datetime_now.month, datetime_now.day,
                    datetime_now.hour, datetime_now.minute, datetime_now.second)

        # Save the Sequential Model
        self.model.save("./models/model_{}_optimiser_{}"
                        .format(self.optimizer_for_model_name.lower(), model_timestamp))

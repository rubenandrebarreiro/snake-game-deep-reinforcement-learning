# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the TensorFlow Library, with tensorflow alias
import tensorflow as tensorflow

# From the TensorFlow.Keras Module, import the Sequential Learning Model
from tensorflow.keras import Sequential

# From the TensorFlow.Keras.Layers Module, import the Convolutional 2D Layer
from tensorflow.keras.layers import Conv2D

# From the TensorFlow.Keras.Layers Module, import the Maximum Pooling 2D Layer
from tensorflow.keras.layers import MaxPooling2D

# From the TensorFlow.Keras.Layers Module, import the Activation Layer
from tensorflow.keras.layers import Activation

# From the TensorFlow.Keras.Layers Module, import the Dense Layer
from tensorflow.keras.layers import Dense

from tensorflow.keras.initializers import HeUniform

from datetime import datetime


class SnakeAgentCNNModel:

    def __init__(self, optimizer_for_model_name, optimizer_for_model,
                 input_shape, hidden_shapes, output_shape):
        self.model = None
        self.optimizer_for_model_name = optimizer_for_model_name
        self.optimizer_for_model = optimizer_for_model
        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape

    def compute_model(self):

        # Initialise the He Uniform Initializer for the Model
        he_uniform_initializer = HeUniform()

        # Initialise the Sequential Model
        self.model = Sequential()

        # Add a 2D Convolutional Layer,
        # with 16 Units, a 8x8 Kernel, a 4x4 Stride and a He Uniform Initializer,
        # for the Shape of the Current State Vector
        self.model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                              kernel_inializer=he_uniform_initializer,
                              input_shape=self.input_shape))

        # Add a Maximum 2D Pooling Layer, with 2x2 Pool Size
        self.model.add(MaxPooling2D((2, 2)))

        # Add a ReLU Activation Layer
        self.model.add(Activation("relu"))

        # Add a 2D Convolutional Layer,
        # with 32 Units, a 4x4 Kernel, a 2x2 Stride and a He Uniform Initializer
        self.model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                              kernel_inializer=he_uniform_initializer))

        # Add a Maximum 2D Pooling Layer, with 2x2 Pool Size
        self.model.add(MaxPooling2D((2, 2)))

        # Add a ReLU Activation Layer, with 2x2 Pool Size
        self.model.add(Activation("relu"))

        # Add a Dense Layer, with 256 Units, with the He Uniform Initializer
        self.model.add(Dense(256, kernel_initializer=he_uniform_initializer))

        # Add a Dense Layer, for the Shape of the Actions Vector
        self.model.add(Dense(self.output_shape))

        # Compile the Sequential Model, with the Huber Loss, using the Adam Optimiser
        self.model.compile(loss=tensorflow.keras.losses.Huber(),
                           optimizer=self.optimizer_for_model,
                           metrics=["accuracy"])

        # Return the Learning Model
        return self.model

    def save_model(self):

        datetime_now = datetime.now()

        model_timestamp = "{}_{}_{}_{}_{}_{}"\
            .format(datetime_now.year, datetime_now.month, datetime_now.day,
                    datetime_now.hour, datetime_now.minute, datetime_now.second)

        self.model.save("./models/model_{}_optimiser_{}"
                        .format(self.optimizer_for_model_name.lower(), model_timestamp))

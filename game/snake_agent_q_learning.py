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

# From the Collections Library, import the Deque Module
from collections import deque

# Import the NumPy, with numpy alias
import numpy as numpy

# Import Random Library, with random alias
import random as random

# Import the Snake Game, with the Snake_Game alias
from snake_game import SnakeGame as Snake_Game

# Set 300 Training Episodes, where one episode corresponds to
# a full "game", until the pole falls or the cart gets too far from the centre
training_episodes = 300


# Function to compute the Learning Mode for the Snake Agent
def compute_model_for_snake_agent(current_state_shape, actions_vector_shape):

    # Set the Learning Rate as 0.001
    learning_rate = 0.001

    # Retrieve the He Uniform Initializer for the Model
    initializer = tensorflow.keras.initializers.HeUniform()

    # Initialise the Sequential Model
    model = Sequential()

    # Add a 2D Convolutional Layer,
    # with 16 Units, a 8x8 Kernel, a 4x4 Stride and a He Uniform Initializer,
    # for the Shape of the Current State Vector
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                     kernel_inializer=initializer, input_shape=current_state_shape))  # TODO - Shape?

    # Add a Maximum 2D Pooling Layer, with 2x2 Pool Size
    model.add(MaxPooling2D((2, 2)))

    # Add a ReLU Activation Layer
    model.add(Activation("relu"))

    # Add a 2D Convolutional Layer,
    # with 32 Units, a 4x4 Kernel, a 2x2 Stride and a He Uniform Initializer
    model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                     kernel_inializer=initializer))

    # Add a Maximum 2D Pooling Layer, with 2x2 Pool Size
    model.add(MaxPooling2D((2, 2)))

    # Add a ReLU Activation Layer, with 2x2 Pool Size
    model.add(Activation("relu"))

    # Add a Dense Layer, with 256 Units, with the He Uniform Initializer
    model.add(Dense(256, kernel_initializer=initializer))

    # Add a Dense Layer, for the Shape of the Actions Vector
    model.add(Dense(actions_vector_shape))

    # Compile the Sequential Model, with the Huber Loss, using the Adam Optimiser
    model.compile(loss=tensorflow.keras.losses.Huber(),
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    # Return the Learning Model
    return model


# Function to train the Learning Model to approximate it to the Target Model
def train(replay_memory, model, target_model):

    # Set the Discount Factor
    discount_factor = 0.618

    # Set the Batch Size
    batch_size = (64 * 2)

    # Sample randomly the Replay Memory Queue, according to the size of the Batch,
    # which acts as a pool of experiences for Training
    mini_batch = random.sample(replay_memory, batch_size)

    # Retrieve the current States of the Transitions
    current_states = numpy.array([transition[0] for transition in mini_batch])

    # Predict the current Q-Values, according to the current states
    current_q_values_list = model.predict(current_states)

    # Set the new current States, from the Transitions in the Mini-Batch
    new_current_states = numpy.array([transition[3] for transition in mini_batch])

    # Predict the future Q-Values, according to the new current States
    future_q_values_list = target_model.predict(new_current_states)

    # Initialise the Observations, i.e., the xs (Features) of the Data
    xs_data = []

    # Initialise the Q-Values to the ys (Targets) of the Data
    ys_data = []

    # For each Mini-Batch, retrieves the respective Experience, from the Memory Queue
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):

        # If the Train is not done yet
        if not done:

            # Compute the new Maximum of Q-Value, for the Future actions,
            # summing it to the current reward
            max_future_q_value = (reward + discount_factor * numpy.max(future_q_values_list[index]))

        # If the Train is already done
        else:

            # Set the current reward as the Maximum of Q-Values,
            # since there are no more actions to take
            max_future_q_value = reward

        # Set the current Q-Values, from the list of the current Q-Values
        current_q_values = current_q_values_list[index]

        # Set the current Q-Values, for each action,
        # according to the Maximum of Q-Values, summed to the current rewards
        current_q_values[action] = max_future_q_value

        # Append the current Observation to the xs (Features) of the Data
        xs_data.append(observation)

        # Append the current Q-Values to the ys (Targets) of the Data
        ys_data.append(current_q_values)

    # Fit the Learning Model, according to the Observations, i.e., the xs (Features) of the Data
    # and to the Q-Values, i.e., the ys (Targets) of the Data
    model.fit(numpy.array(xs_data), numpy.array(ys_data), batch_size=batch_size, verbose=0, shuffle=True)


# Execute the Snake Agent
def execute_snake_agent(snake_game):

    # The Threshold for the Random Number generated
    epsilon = 1

    # The Maximum value for the Threshold for the Random Number generated
    max_epsilon = 1

    # The Minimum value for the Threshold for the Random Number generated
    min_epsilon = 0.01

    # The Decay factor
    decay = 0.01

    # The minimum size for the Replay Memory
    min_replay_memory_size = 1000

    # Build the Learning Model for the Learning Agent,
    # according to the space of the CartPole Environment
    model = compute_model_for_snake_agent(environment.observation_space.shape, environment.action_space.n)

    # Build the Learning Model for the Target Agent,
    # according to the space of the CartPole Environment
    target_model = compute_model_for_snake_agent(environment.observation_space.shape, environment.action_space.n)

    # Set the Weights of the Learning Model for the Target Agent,
    # according to its Weights
    target_model.set_weights(model.get_weights())

    # Build the Replay Memory
    replay_memory = deque(maxlen=50000)

    # Initialise the Steps for updating the Learning Model
    steps_to_update_target_model = 0

    # For each Training Episode
    for episode in range(training_episodes):

        # Initialise the Total Training Rewards, for the current Training Episode
        total_training_rewards = 0

        # Initialise the Observation, for the current Training Episode, resetting it
        observation = snake_game.reset()

        # Initialise the Boolean Flag of the information about
        # the Training is done or not
        done = False

        # While the Training
        while not done:

            # Increment the Steps to update the Target Model
            steps_to_update_target_model += 1

            # Dummy Boolean Flag
            # (it is always True, it can be changed)
            if True:

                # Render the Environment
                environment.render()

            # Generate a Random Number
            random_number = numpy.random.rand()

            # If the Random Number generated is lower or equal to
            # the Threshold for the Random Number generated
            if random_number <= epsilon:

                # Sample an action for a possible action
                action = environment.action_space.sample()

            # If the Random Number generated is greater than
            # the Threshold for the Random Number generated
            else:

                # Reshape the Observations
                reshaped = observation.reshape([1, observation.shape[0]])

                # Predict the possible next Q-Values (Rewards)
                predicted = model.predict(reshaped).flatten()

                # Save the action that maximizes the Q-Value (Reward)
                action = numpy.argmax(predicted)

            # Run the action with maximum Q-Value (Reward),
            # as one timestep on the dynamics of the Environment
            new_observation, reward, done, info = snake_game.step(action)

            # Append the last Experience to the Replay Memory
            replay_memory.append([observation, action, reward, new_observation, done])

            # If the Replay Memory have a size greater than the minimum value for it,
            # and, the number of Steps to update the Target Model is multiple of 4
            if len(replay_memory) >= min_replay_memory_size and \
                    (steps_to_update_target_model % 4 == 0 or done):

                # Train the Learning Model
                train(replay_memory, model, target_model)

            # Update the current Observation
            observation = new_observation

            # Sum the current Reward to the Total Training Rewards,
            # for the current Training Episode
            total_training_rewards += reward

            # If the Training is done
            if done:

                # Print the debug information of the Total Rewards already achieved
                print("Rewards: {} after n steps/episodes = {}, with final reward = {}\n"
                      .format(total_training_rewards, episode, reward))

                # Increment the Total Training Rewards
                total_training_rewards += 1

                # After 100 steps, the Target Model will be updated
                if steps_to_update_target_model >= 100:
                    # Print the debug information about the updating of the Target Model
                    print("\nCopying the Main Network's Weights to the Target Network's Weights...\n\n")

                    # Set the Weights of the Target Model
                    target_model.set_weights(model.get_weights())

                    # Reset the number of Steps for the updating of the Target Model
                    steps_to_update_target_model = 0

                # Break the loop, when the Training is done
                break

            # Update the Threshold for the Random Number generated,
            # according to the Minimum and Maximum values for it, as also, to the Decay factor
            epsilon = (min_epsilon + ((max_epsilon - min_epsilon) * numpy.exp(-decay * episode)))


# Runnable Method
if __name__ == "__main__":

    # Create the Snake Game, for a Board Game of (30x30)
    snake_game = Snake_Game(30, 30, border=1)

    # Reset the Snake Game
    board, reward, done, info = snake_game.reset()

    execute_snake_agent(snake_game)

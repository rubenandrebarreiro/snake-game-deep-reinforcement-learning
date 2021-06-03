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

# From the TensorFlow.Keras.Optimizers Module,
# import the Stochastic Gradient Descent (SGD) Optimizer
from tensorflow.keras.optimizers import SGD

# From the TensorFlow.Keras.Optimizers Module,
# import the Adam Optimizer
from tensorflow.keras.optimizers import Adam

# From the TensorFlow.Keras.Optimizers Module,
# import the RMSProp Optimizer
from tensorflow.keras.optimizers import RMSprop

# From the TensorFlow.Keras.Optimizers Module,
# import the ADAMax Optimizer
from tensorflow.keras.optimizers import Adamax

# From the TensorFlow.Keras.Optimizers Module,
# import the ADAGrad Optimizer
from tensorflow.keras.optimizers import Adagrad

# From the TensorFlow.Keras.Optimizers Module,
# import the ADADelta Optimizer
from tensorflow.keras.optimizers import Adadelta

# From the TensorFlow.Keras.Losses Module,
# import the Mean Squared Error (MSE) Optimizer
from tensorflow.keras.losses import MSE

# From the NumPy Library, import the Max function
from numpy import max

# From the NumPy Library, import the Array function
from numpy import array

# From the Game.Others.Parameters_Arguments Module,
# import the size of the Batch
from game.others.parameters_arguments import BATCH_SIZE


# Class for the Snake Agent's Q-Learning Trainer
class SnakeAgentQLearningTrainer:

    # Constructor of the Snake Agent's Q-Learning Trainer
    def __init__(self, learning_rate, gamma_discount_factor):

        # Initialise the CNN (Convolutional Neural Network) Model for the current observations, as None
        self.snake_cnn_model_for_current_observations = None

        # Initialise the CNN (Convolutional Neural Network) Model for the target observations, as None
        self.snake_cnn_model_for_target_observations = None

        # Set the Learning Rate for the Snake Agent's Q-Learning Trainer
        self.learning_rate = learning_rate

        # Set the Gamma value (Discount Factor) for the Snake Agent's Q-Learning Trainer
        self.gamma_discount_factor = gamma_discount_factor

        # Set the Optimiser for the Snake Agent's Q-Learning Trainer
        self.optimizer = Adam(learning_rate=learning_rate)

    # Initialise the final CNN (Convolutional Neural Network) Models for the current and target observations
    def initialise_cnn_models(self, snake_cnn_model_for_current_observations,
                              snake_cnn_model_for_target_observations):

        # Set the final CNN (Convolutional Neural Network) Model for the current observations
        self.snake_cnn_model_for_current_observations = \
            snake_cnn_model_for_current_observations

        # Set the final CNN (Convolutional Neural Network) Model for the target observations
        self.snake_cnn_model_for_target_observations = \
            snake_cnn_model_for_target_observations

        # Compute the final CNN (Convolutional Neural Network) Model for the current observations
        self.snake_cnn_model_for_current_observations.compute_model()

        # Compute the final CNN (Convolutional Neural Network) Model for the target observations
        self.snake_cnn_model_for_target_observations.compute_model()

    # Function for the Snake Agent's Q-Learning Trainer train a Step
    def train_step(self, observations, actions, rewards, new_observations, dones):

        # If the dones is a tuple of values
        if isinstance(dones, tuple):

            # Retrieve the number of Experience's Examples
            num_experience_examples = len(dones)

            # Convert the observations for an NumPy Array,
            # as the Current States of the Snake Agent
            current_states = array(observations)

            # Convert the new observations for an NumPy Array,
            # as the New States of the Snake Agent
            new_states = array(new_observations)

        # If the dones is a single value
        else:

            # Retrieve the number of Experience's Examples, as 1
            num_experience_examples = 1

            # Convert the observations for an NumPy Array,
            # as the Current States of the Snake Agent
            current_states = array([observations])

            # Convert the new observations for an NumPy Array,
            # as the New States of the Snake Agent
            new_states = array([new_observations])

        # Predict the Q-Values, according to the Current States of the Snake Agent
        q_values_list_for_current_states = \
            self.snake_cnn_model_for_current_observations.model.predict(current_states)

        # Predict the Q-Values, according to the New States of the Snake Agent
        q_values_list_for_new_states = \
            self.snake_cnn_model_for_target_observations.model.predict(new_states)

        # Initialise the current Observations as the xs (Features) of the Data
        xs_features_data = []

        # Initialise the Q-Values for the new Observations as the ys (Targets) of the Data
        ys_targets_data = []

        # For each Experience's Examples
        for index_experience_example in range(num_experience_examples):

            # If the dones is a list of values
            if isinstance(dones, tuple):

                # Retrieve the Observation for the current Experience's Example
                observation = observations[index_experience_example]

                # Retrieve the Reward for the current Experience's Example
                reward = rewards[index_experience_example]

                # Retrieve the Action for the current Experience's Example
                action = actions[index_experience_example]

                # Retrieve the Done Flag for the current Experience's Example
                done = dones[index_experience_example]

                # Set the current Q-Values, from the list of the current Q-Values
                current_q_values = q_values_list_for_current_states[index_experience_example]

            # If the dones is a single value
            else:

                # Retrieve the Observation for the current Experience's Example
                observation = observations

                # Retrieve the Reward for the current Experience's Example
                reward = rewards

                # Retrieve the Action for the current Experience's Example
                action = actions

                # Retrieve the Done Flag for the current Experience's Example
                done = dones

                # Set the current Q-Values, from the list of the current Q-Values
                current_q_values = q_values_list_for_current_states

            # If the Train is not done for the current Experience's Example
            if not done:

                # Compute the new Maximum of Q-Value, for the Future actions,
                # summing it to the current reward
                max_future_action_q_value = self.compute_q_new_value_update_rule(
                    reward, self.gamma_discount_factor,
                    q_values_list_for_new_states[index_experience_example])

            # If the Train is already done for the current Experience's Example
            else:

                # Set the current reward as the Maximum of Q-Values,
                # since there are no more actions to take
                max_future_action_q_value = reward

            # If the dones is a list of values
            if isinstance(dones, tuple):

                # Reshape the current Q-Values, for each Action
                current_q_values = current_q_values.reshape(-1)

                # Set the current Q-Values, for each Action,
                # according to the Maximum of Q-Values, summed to the current rewards
                current_q_values[(action + 1)] = max_future_action_q_value

            # If the dones is a single value
            else:

                # Set the current Q-Values, for each action,
                # according to the Maximum of Q-Values, summed to the current rewards
                current_q_values[0] = max_future_action_q_value

            # Append the current Observation to the xs (Features) of the Data
            xs_features_data.append(observation)

            # Append the current Q-Values to the ys (Targets) of the Data
            ys_targets_data.append(current_q_values)

        # Fit the CNN (Convolutional Neural Network) Model,
        # according to the current Observations, i.e., the xs (Features) of the Data
        # and to the Q-Values of the new Observations, i.e., the ys (Targets) of the Data
        self.snake_cnn_model_for_current_observations\
            .model.fit(array(xs_features_data), array(ys_targets_data),
                       batch_size=BATCH_SIZE, verbose=0, shuffle=True)

    # Static function to compute the new Q-Value, for the given Q-Values of the new observations,
    # following the update rule for the reward
    @staticmethod
    def compute_q_new_value_update_rule(reward, gamma_discount_factor, q_values_new_observation):
        return reward + gamma_discount_factor * max(q_values_new_observation)

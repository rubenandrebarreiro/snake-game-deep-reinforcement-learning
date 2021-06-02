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

from numpy import max


class SnakeAgentQLearningTrainer:

    def __init__(self, model, learning_rate, gamma_discount_factor):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma_discount_factor = gamma_discount_factor
        self.optimizer = Adam(learning_rate=learning_rate)
        self.criteria_error = MSE()

    def train_step(self, observations, actions, rewards, new_observations, dones):
        pass

    def compute_q_new_value_update_rule(self, reward, gamma_discount_factor, q_values_new_observation):
        return reward + gamma_discount_factor * max(q_values_new_observation)

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

# Import the Multi-Processing Python's Module as multiprocessing alias
import multiprocessing as multiprocessing

# Import the Tensorflow as Tensorflow alias Python's Module
import tensorflow as tensorflow


# Constants

# The boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs)
TENSORFLOW_KERAS_HPC_BACKEND_SESSION = True

# The Number of CPU's Processors/Cores
NUM_CPU_PROCESSORS_CORES = multiprocessing.cpu_count()

# The Number of GPU's Devices
NUM_GPU_DEVICES = len(tensorflow.config.list_physical_devices('GPU'))

# The Maximum Memory capacity for the Snake Agent
MAX_MEMORY = 100000

# The initial value for the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
INITIAL_EPSILON_RANDOMNESS = 1

# The Maximum value for the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
MAXIMUM_EPSILON_RANDOMNESS = 1

# The Minimum value for the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
MINIMUM_EPSILON_RANDOMNESS = 0.01

# The Decay Factor to adjust the value for
# the Epsilon variable for the Randomness used to,
# decide about Exploration and Exploitation
DECAY_FACTOR_EPSILON_RANDOMNESS = 0.01

# The Gamma (Discount Reward) for the Q-Learning Algorithm
GAMMA_DISCOUNT_FACTOR = 0.9

# The Number of Games (Training Episodes)
NUM_GAME_TRAINING_EPISODES = 300

# The size of the Batch of Examples of Observations
BATCH_SIZE = 1000

# The List of Optimisers available to use for
# the CNN (Convolutional Neural Network) Model
AVAILABLE_OPTIMISERS_LIST = ["SGD", "RMSPROP", "ADAM", "ADAGRAD", "ADADELTA", "ADAMAX"]

# The Number of Optimisers available to use for
# the CNN (Convolutional Neural Network) Model
NUM_AVAILABLE_OPTIMISERS = len(AVAILABLE_OPTIMISERS_LIST)

# The Learning Rates for the Optimisers used for
# the CNN (Convolutional Neural Network) Model
INITIAL_LEARNING_RATES = [0.005, 0.0005, 0.001, 0.012, 0.25, 0.001]

# The List of Kernel Sizes to use,
# in the CNN (Convolutional Neural Network) Model
KERNEL_SIZES_LIST = [8, 4]

# The List of Strides to use,
# in the CNN (Convolutional Neural Network) Model
STRIDES_LIST = [4, 2]

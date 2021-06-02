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

# Constants and Flags

# The Maximum Memory capacity for the Snake Agent
MAX_MEMORY = 100000

# The size of the Batch of Examples of Observations
BATCH_SIZE = 1000

# The Learning Rate for the CNN (Convolutional Neural Network) Model
LEARNING_RATE = 0.001

# The List of Optimisers to use,
# in the CNN (Convolutional Neural Network) Model
OPTIMISERS_LIST = ["ADAM"]

# The List of Kernel Sizes to use,
# in the CNN (Convolutional Neural Network) Model
KERNEL_SIZES_LIST = [8, 4]

# The List of Strides to use,
# in the CNN (Convolutional Neural Network) Model
STRIDES_LIST = [4, 2]

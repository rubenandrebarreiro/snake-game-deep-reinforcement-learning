from tensorflow.keras.optimizers import Adam

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

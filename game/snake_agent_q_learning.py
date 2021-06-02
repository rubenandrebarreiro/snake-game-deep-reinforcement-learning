# From the Collections Library, import the Deque Module
from collections import deque

# Import the NumPy, with numpy alias
import numpy as numpy

# Set 300 Training Episodes, where one episode corresponds to
# a full "game", until the pole falls or the cart gets too far from the centre
training_episodes = 300


# Execute the Snake Agent TODO - To be deleted soon
def execute_snake_agent(snake_game_environment):

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
    model = compute_model_for_snake_agent((snake_game_environment.height, snake_game_environment.width), 3)  # TODO - Ver action space - 3 actions ?? (go straight, turn left, turn right)

    # Build the Learning Model for the Target Agent,
    # according to the space of the CartPole Environment
    target_model = compute_model_for_snake_agent((snake_game_environment.height, snake_game_environment.width), 3)  # TODO - Ver action space - 3 actions ?? (go straight, turn left, turn right)

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
        observation = snake_game_environment.reset()

        # Initialise the Boolean Flag of the information about
        # the Training is done or not
        game_done = False

        # While the Training
        while not game_done:

            # Increment the Steps to update the Target Model
            steps_to_update_target_model += 1

            # Generate a Random Number
            random_number = numpy.random.rand()

            # If the Random Number generated is lower or equal to
            # the Threshold for the Random Number generated
            if random_number <= epsilon:

                # Sample a possible Action
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
            new_observation, reward, done, info = snake_game_environment.step(action)

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

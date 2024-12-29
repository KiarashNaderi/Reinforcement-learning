import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    # Environment setup without rendering
    if render:
        env = gym.make('Taxi-v3', render_mode='human')
    else:
        env = gym.make('Taxi-v3')

    # Load or initialize Q-table
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # init Q-table
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    # Hyperparameters
    learning_rate_a = 0.9  # Alpha or learning rate
    discount_factor_g = 0.9  # Gamma or discount rate
    epsilon = 1  # 100% random actions initially
    epsilon_decay_rate = 0.0001  # Epsilon decay rate
    rng = np.random.default_rng()  # Random number generator

    # Rewards storage
    rewards_per_episode = np.zeros(episodes)

    # Q-learning algorithm
    for i in range(episodes):
        state = env.reset()[0]  # Initial state
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(q[state, :])  # Best action from Q-table

            # Execute action and get feedback
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

            if reward == 20:  # Adjust reward condition based on the environment
                rewards_per_episode[i] = 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)  # Decay epsilon

    # Plot rewards over episodes
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig('taxi.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

    # Save Q-table if in training mode
    if is_training:
        with open('taxi.pkl', 'wb') as f:
            pickle.dump(q, f)

    env.close()

if __name__ == "__main__":
    # Run the training or evaluation
    run(1000, is_training=True, render=False)  # Training mode without rendering

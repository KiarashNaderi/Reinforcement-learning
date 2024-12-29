import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, render=False):

    # Environment setup
    if render:
        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human')
    else:
        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)

    # Initialize Q-table with zeros
    q = np.zeros((env.observation_space.n, env.action_space.n))

    # Hyperparameters
    learning_rate_a = 0.9  # Alpha or learning rate
    discount_factor_g = 0.9  # Gamma or discount factor
    epsilon = 1  # 100% random actions initially
    epsilon_decay_rate = 0.0001  # Epsilon decay rate
    rng = np.random.default_rng()  # Random number generator

    # Rewards storage
    rewards_per_episode = np.zeros(episodes)
    cumulative_rewards = np.zeros(episodes)  # Array to store cumulative rewards

    # Q-learning algorithm
    for i in range(episodes):
        state = env.reset()[0]  # Initial state
        terminated = False
        truncated = False
        episode_reward = 0  # Initialize reward for the current episode

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(q[state, :])  # Choose action with highest Q-value

            # Execute action and get feedback from environment
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update rule
            q[state, action] = q[state, action] + learning_rate_a * (
                reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )

            state = new_state
            episode_reward += reward  # Add reward of the current step to episode reward

        rewards_per_episode[i] = episode_reward  # Store total reward for the episode
        if i > 0:
            cumulative_rewards[i] = cumulative_rewards[i - 1] + episode_reward  # Calculate cumulative rewards

        epsilon = max(epsilon - epsilon_decay_rate, 0)  # Decay epsilon over time

    # Plot cumulative rewards over episodes
    plt.plot(np.arange(episodes), cumulative_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.savefig('frozen_lake8x8_cumulative.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

    # Save Q-table
    with open('frozen_lake8x8.pkl', 'wb') as f:
        pickle.dump(q, f)

    env.close()  # Make sure the environment is closed after the simulation

if __name__ == "__main__":
    run(15000)  # Run for 15000 episodes

import gymnasium as gym
import numpy as np
import pickle

def play_using_trained_model(episodes=5, render=True):

    # Load the trained Q-table from file
    with open('taxi.pkl', 'rb') as f:
        q = pickle.load(f)

    # Setup the environment
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    for i in range(episodes):
        state = env.reset()[0]  # Reset the environment
        terminated = False
        truncated = False
        total_reward = 0  # To store total reward of each episode

        print(f"Episode {i+1}:")

        while not terminated and not truncated:
            if render:
                env.render()  # Render the environment if needed

            # Select the best action using the trained Q-table
            action = np.argmax(q[state, :])

            # Execute the action and get feedback
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward  # Add the reward to total reward

            state = new_state

        print(f"Total reward in episode {i+1}: {total_reward}")
    
    env.close()  # Close the environment after the episodes are done

if __name__ == "__main__":
    play_using_trained_model(episodes=5, render=True)  # Play 5 episodes with rendering

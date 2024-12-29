import gymnasium as gym
import numpy as np
import pickle

def play_using_trained_model(episodes=10, render=True):
    # بارگذاری Q-table ذخیره شده
    with open('frozen_lake8x8.pkl', 'rb') as f:
        q = pickle.load(f)

    # تنظیم محیط
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    for i in range(episodes):
        state = env.reset()[0]  # بازنشانی وضعیت اولیه
        terminated = False
        truncated = False
        total_reward = 0  # پاداش تجمعی برای هر اپیزود

        print(f"Episode {i+1}:")

        while not terminated and not truncated:
            if render:
                env.render()  # نمایش محیط

            # انتخاب عمل بر اساس Q-table
            action = np.argmax(q[state, :])

            # اجرای عمل و دریافت وضعیت جدید و پاداش
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            state = new_state

        print(f"Total reward in episode {i+1}: {total_reward}")
    
    env.close()  # بستن محیط بعد از بازی

if __name__ == "__main__":
    play_using_trained_model(episodes=5, render=True)  # بازی با مدل آموزش‌دیده برای ۵ اپیزود

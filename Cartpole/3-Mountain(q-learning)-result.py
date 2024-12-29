import gymnasium as gym
import numpy as np
import pickle

def play_using_trained_model(episodes=5, render=True):

    # بارگذاری مدل Q-table آموزش‌دیده از فایل
    with open('mountain_car.pkl', 'rb') as f:
        q = pickle.load(f)

    # تنظیم محیط بازی
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # تقسیم فضاهای موقعیت و سرعت برای استفاده از Q-table
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    for i in range(episodes):
        state = env.reset()[0]  # ریست کردن محیط
        state_p = np.digitize(state[0], pos_space)  # تبدیل موقعیت به ایندکس Q-table
        state_v = np.digitize(state[1], vel_space)  # تبدیل سرعت به ایندکس Q-table

        terminated = False
        total_reward = 0  # جمع کل پاداش برای هر اپیزود

        print(f"Episode {i+1}:")

        while not terminated:
            if render:
                env.render()  # نمایش محیط بازی

            # انتخاب بهترین عمل از Q-table آموزش‌دیده
            action = np.argmax(q[state_p, state_v, :])

            # اجرای عمل و دریافت وضعیت جدید و پاداش
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            total_reward += reward  # جمع پاداش

            state_p = new_state_p
            state_v = new_state_v

        print(f"Total reward in episode {i+1}: {total_reward}")
    
    env.close()  # بستن محیط پس از اجرای بازی

if __name__ == "__main__":
    play_using_trained_model(episodes=5, render=True)  # اجرای 5 اپیزود با رندر

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    # ساخت محیط MountainCar
    if render:
        env = gym.make('MountainCar-v0', render_mode='human')
    else:
        env = gym.make('MountainCar-v0')

    # تقسیم فضاهای موقعیت و سرعت برای Q-table
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    # بارگذاری یا مقداردهی اولیه Q-table
    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))  # Q-table با اندازه مناسب
    else:
        with open('mountain_car.pkl', 'rb') as f:
            q = pickle.load(f)

    # پارامترهای الگوریتم
    learning_rate_a = 0.9  # نرخ یادگیری
    discount_factor_g = 0.9  # ضریب تخفیف
    epsilon = 1  # شروع با 100% انتخاب تصادفی
    epsilon_decay_rate = 2 / episodes  # نرخ کاهش epsilon
    rng = np.random.default_rng()  # تولیدکننده اعداد تصادفی

    # ذخیره پاداش‌های اپیزودها
    rewards_per_episode = np.zeros(episodes)

    # الگوریتم Q-learning
    for i in range(episodes):
        state = env.reset()[0]  # موقعیت اولیه
        state_p = np.digitize(state[0], pos_space)  # تبدیل موقعیت به ایندکس Q-table
        state_v = np.digitize(state[1], vel_space)  # تبدیل سرعت به ایندکس Q-table

        terminated = False
        rewards = 0  # پاداش اپیزود

        while not terminated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # انتخاب تصادفی در حالت آموزش
            else:
                action = np.argmax(q[state_p, state_v, :])  # بهترین عمل از Q-table

            # اجرای عمل و دریافت وضعیت جدید و پاداش
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            rewards += reward

            # به‌روزرسانی Q-table در حالت آموزش
            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )

            state_p = new_state_p
            state_v = new_state_v

        # کاهش epsilon با گذشت زمان
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # ذخیره پاداش اپیزود
        rewards_per_episode[i] = rewards

    # رسم نمودار میانگین پاداش‌ها در اپیزودها
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(mean_rewards)
    plt.savefig('mountain_car.png')  # ذخیره نمودار
    plt.show()  # نمایش نمودار

    # ذخیره Q-table پس از آموزش
    if is_training:
        with open('mountain_car.pkl', 'wb') as f:
            pickle.dump(q, f)

    env.close()

if __name__ == "__main__":
    # اجرای آموزش بدون رندر
    run(1000, is_training=True, render=False)  # آموزش 1000 اپیزود بدون رندر

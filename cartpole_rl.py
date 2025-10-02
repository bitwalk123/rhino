import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # ログフォルダの準備
    dir_log = "./logs/"
    os.makedirs(dir_log, exist_ok=True)

    # 学習環境の準備
    env = gym.make("CartPole-v1", render_mode="human")
    env = Monitor(env, dir_log)  # Monitorの利用

    # モデルの準備
    model = PPO("MlpPolicy", env, verbose=True)

    # 学習の実行
    model.learn(total_timesteps=50000)

    # 推論の実行
    obs, info = env.reset()
    print(f"Starting observation: {obs}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

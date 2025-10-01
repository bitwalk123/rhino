import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    # モデルの準備
    model = PPO('MlpPolicy', env, verbose=True)

    # 学習の実行
    #model.learn(total_timesteps=128000)
    model.learn(total_timesteps=50000)

    obs, info = env.reset()
    print(f"Starting observation: {obs}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        #action = env.action_space.sample()  # Random action for now - real agents will be smarter!
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()
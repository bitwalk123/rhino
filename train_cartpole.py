import gymnasium as gym
from stable_baselines3 import PPO

# 学習環境の準備
env = gym.make('CartPole-v1', render_mode="rgb_array")

# モデルの準備
model = PPO('MlpPolicy', env, verbose=1)

# 学習の実行
model.learn(total_timesteps=128000)

# 推論の実行
obs, info = env.reset()
while True:
    # 学習環境の描画
    env.render()

    # モデルの推論
    action, _ = model.predict(obs, deterministic=True)

    # 1ステップ実行
    obs, reward, done, truncated, info = env.step(action)

    # エピソード完了
    if done:
        break

# 学習環境の解放
env.close()

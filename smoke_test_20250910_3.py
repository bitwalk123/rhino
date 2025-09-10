import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import TradingEnv  # TradingEnv を保存したファイルに合わせて変更

# ===== ダミーデータ作成 =====
n_ticks = 2000
df = pd.DataFrame({
    "Time": np.arange(n_ticks),
    "Price": np.cumsum(np.random.randn(n_ticks)) + 100.0,
    "Volume": np.random.randint(1, 100, size=n_ticks)
})

# ===== パラメータ =====
N_EPISODES = 100

# 集計用
episode_rewards = []
final_pnls = []

for ep in range(N_EPISODES):
    env = TradingEnv(df)
    obs, info = env.reset()
    done = False
    rewards = []
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
    episode_rewards.append(np.sum(rewards))
    final_pnls.append(info["pnl_total"])

print(f"Ran {N_EPISODES} episodes")
print(f"Avg total reward = {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Avg final PnL    = {np.mean(final_pnls):.2f} ± {np.std(final_pnls):.2f}")

# ===== 可視化 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 報酬分布
axes[0].hist(episode_rewards, bins=30, alpha=0.7)
axes[0].set_title("Total reward per episode")
axes[0].set_xlabel("Total reward")
axes[0].set_ylabel("Frequency")

# PnL 分布
axes[1].hist(final_pnls, bins=30, alpha=0.7)
axes[1].set_title("Final PnL per episode")
axes[1].set_xlabel("PnL")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

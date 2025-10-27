import os

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from funcs.ios import get_excel_sheet
from modules.env import TrainingEnv
from structs.res import AppRes


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_mask = info.get("action_mask", np.ones(self.env.action_space.n, dtype=np.int8))
        return obs, info

    def step(self, action):
        if self.action_mask[action] == 0:
            # 無効なアクションを選んだ場合、強制的に HOLD に置き換える
            action = 0  # ActionType.HOLD.value
        obs, reward, done, truncated, info = self.env.step(action)
        self.action_mask = info.get("action_mask", np.ones(self.env.action_space.n, dtype=np.int8))
        return obs, reward, done, truncated, info


if __name__ == "__main__":
    res = AppRes()

    # 学習用データフレーム
    code = "7011"
    file = "ticks_20250819.xlsx"
    path_excel = os.path.join(res.dir_collection, file)
    df = get_excel_sheet(path_excel, code)

    # df: ティックデータ（Time, Price, Volume）を含む DataFrame
    env_raw = TrainingEnv(df)
    env = ActionMaskWrapper(env_raw)

    # SB3の環境チェック（オプション）
    check_env(env, warn=True)

    # PPOエージェントの定義と学習
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

    df_transaction = pd.DataFrame(env_raw.trans_man.dict_transaction)
    print(df_transaction)
    print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from funcs.ios import get_excel_sheet
from modules.agent_auxiliary import ActionMaskWrapper
from modules.env import TrainingEnv
from structs.res import AppRes

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

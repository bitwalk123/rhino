import os

import matplotlib.pyplot as plt
import pandas as pd

from funcs.ios import get_excel_sheet
from funcs.models import get_trained_ppo_model_path
from modules.agent import PPOAgent
from structs.res import AppRes


def plot_reward_distribution(ser: pd.Series):
    plt.hist(ser, bins=100)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    res = AppRes()

    n_epoch = 10
    flag_new_model = True

    # PPO エージェントのインスタンス
    agent = PPOAgent()

    # 学習用データフレーム
    code = "7011"
    # list_file = sorted(os.listdir(res.dir_collection))
    list_file = ["ticks_20250819.xlsx"]
    for idx, file in enumerate(list_file):
        path_excel = os.path.join(res.dir_collection, file)
        df = get_excel_sheet(path_excel, code)

        # モデルの保存先
        model_path = get_trained_ppo_model_path(res, code)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 学習
        print(f"{idx + 1:>4d}/{len(list_file):>4d}: {file}")
        agent.train(
            df,
            model_path,
            num_epoch=n_epoch,
            new_model=flag_new_model
        )
        if flag_new_model:
            flag_new_model = False

    plot_reward_distribution(pd.Series(agent.epoch_log["reward_raw"]))

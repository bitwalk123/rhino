import os

import matplotlib.pyplot as plt
import pandas as pd

from funcs.ios import get_excel_sheet
from funcs.models import get_trained_ppo_model_path
from modules.agent import PPOAgent
from structs.res import AppRes


def plot_reward_distribution(ser: pd.Series):
    plt.hist(ser, bins=20)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid()
    plt.show()


def plot_obs_trend(df: pd.DataFrame, n: int):
    fig = plt.figure(figsize=(15, 9))
    ax = dict()
    gs = fig.add_gridspec(n, 1, wspace=0.0, hspace=0.0)
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    for i in range(n):
        ax[i].plot(df[i])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = AppRes()

    n_epoch = 100
    flag_new_model = True

    # PPO エージェントのインスタンス
    agent = PPOAgent()

    # 学習用データフレーム
    code = "7011"
    list_file = sorted(os.listdir(res.dir_collection))
    # list_file = ["ticks_20250819.xlsx"]
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

    # 取引結果
    df_transaction = agent.get_transaction()
    print(df_transaction)
    print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

    # 報酬分布
    ser_reward = pd.Series(agent.epoch_log["reward_raw"])
    print(
        f"n: {len(ser_reward)}, "
        f"mean: {ser_reward.mean():.3f}, "
        f"stdev: {ser_reward.std():.3f}"
    )
    plot_reward_distribution(ser_reward)

    """
    # 観測空間
    df_obs = pd.concat([pd.Series(row) for row in agent.epoch_log["obs_raw"]], axis=1).T
    rows = df_obs.shape[1]
    plot_obs_trend(df_obs, rows)
    """

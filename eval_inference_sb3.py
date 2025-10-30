import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path
from modules.agent import PPOAgentSB3
from structs.res import AppRes

FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
fm.fontManager.addfont(FONT_PATH)

# FontPropertiesオブジェクト生成（名前の取得のため）
font_prop = fm.FontProperties(fname=FONT_PATH)
font_prop.get_name()

plt.rcParams["font.family"] = font_prop.get_name()


def plot_reward_distribution(ser: pd.Series):
    plt.hist(ser, bins=20)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid()
    plt.show()


def plot_obs_trend(df: pd.DataFrame, n: int, list_ylabel: list, logscale: bool = False):
    fig = plt.figure(figsize=(15, 9))
    ax = dict()
    gs = fig.add_gridspec(n, 1, wspace=0.0, hspace=0.0)
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    for i in range(n):
        ax[i].plot(df[i])
        if i < n - 3:
            ax[i].set_ylim(-1.1, 1.1)
        else:
            ax[i].set_ylim(0, 1.1)
        ax[i].set_ylabel(list_ylabel[i])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3()

    # 推論用データ
    file = "ticks_20250919.xlsx"
    code = "7011"

    print(f"過去データ {file} の銘柄 {code} について推論します。")
    # Excel ファイルのフルパス
    path_excel = get_collection_path(res, file)
    # Excel ファイルをデータフレームに読み込む
    df = get_excel_sheet(path_excel, code)
    path_model = get_ppo_model_path(res, code)

    agent.infer(df, path_model)

    # 取引結果
    df_transaction: pd.DataFrame = agent.results["transaction"]
    print(df_transaction)
    print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

    print("モデルへの報酬分布")
    ser_reward = pd.Series(agent.results["reward"])
    print(
        f"n: {len(ser_reward)}, "
        f"mean: {ser_reward.mean():.3f}, "
        f"stdev: {ser_reward.std():.3f}"
    )
    # 報酬分布
    plot_reward_distribution(ser_reward)
    # 観測値トレンド
    df_obs = pd.concat([pd.Series(row) for row in agent.results["obs"]], axis=1).T
    rows = df_obs.shape[1]
    print(f"観測数 : {rows}")
    list_name = [
        "株価比",
        "株価Δ",
        "MAΔ",
        "ROC",
        "RSI",
        "含損益",
        "NONE",
        "LONG",
        "SHORT"
    ]
    plot_obs_trend(df_obs, rows, list_name, logscale=True)

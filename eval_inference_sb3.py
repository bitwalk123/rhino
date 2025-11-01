from matplotlib import dates as mdates
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


def plot_bar_profit(df: pd.DataFrame):
    df.index = pd.to_datetime(df["注文日時"])
    df.index.name = "DateTime"
    ser = df["損益"].dropna()
    total = int(ser.sum())
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(ser.index, ser, width=0.0005, label=f"総収益 : {total:d} 円")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("確定損益（円/株）")
    ax.grid(axis="y")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_obs_trend(df: pd.DataFrame, n: int, list_ylabel: list):
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


def plot_reward_distribution(ser: pd.Series, logscale: bool = False):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(ser, bins=20)
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward")

    if logscale:
        ax.set_yscale("log")
        ax.set_ylabel("Freq. in Log scale")
    else:
        ax.set_ylabel("Frequency")

    x_low, x_high = ax.get_xlim()
    if -1.0 < x_low:
        x_low = -1.0
    if x_high < 1.0:
        x_high = 1.0
    ax.set_xlim(x_low, x_high)
    ax.grid()

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
    plot_bar_profit(df_transaction)

    print("モデルへの報酬分布")
    ser_reward = pd.Series(agent.results["reward"])
    print(
        f"n: {len(ser_reward)}, "
        f"mean: {ser_reward.mean():.3f}, "
        f"stdev: {ser_reward.std():.3f}"
    )
    # 報酬分布
    plot_reward_distribution(ser_reward, logscale=True)
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
    plot_obs_trend(df_obs, rows, list_name)

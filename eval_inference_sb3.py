import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

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
    plt.grid()
    plt.show()


if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3(res)

    # 推論用データ
    file = "ticks_20250919.xlsx"
    code = "7011"

    print(f"過去データ {file} の銘柄 {code} について推論します。")
    agent.infer(file, code)

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
    plot_reward_distribution(ser_reward)

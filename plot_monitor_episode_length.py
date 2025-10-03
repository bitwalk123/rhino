import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # monitor.csv の読み込み
    dir_log = "./logs"
    name_log = "monitor.csv"
    # 最初の行の読み込みを除外
    df = pd.read_csv(os.path.join(dir_log, name_log), skiprows=[0])

    # エピソードの長さのプロット
    plt.plot(df["l"])
    plt.xlabel("episode")
    plt.ylabel("episode length")
    plt.grid()
    plt.tight_layout()
    plt.show()

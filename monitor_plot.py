import os
import pandas as pd
import matplotlib.pyplot as plt

# monitor.csv の読み込み
dir_log = "./logs"
name_log = "monitor.csv"
# 最初の行の読み込みを除外
df = pd.read_csv(os.path.join(dir_log, name_log), skiprows=[0])

# 報酬のプロット
plt.plot(df["r"])
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.tight_layout()
plt.show()

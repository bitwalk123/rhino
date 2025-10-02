import os
import pandas as pd
import matplotlib.pyplot as plt

# monitor.csvの読み込み
dir_log = "./logs/"
name_log = "monitor.csv"
df = pd.read_csv(os.path.join(dir_log, name_log), skiprows=[0])
print(df)

# 報酬のプロット
#x = range(len(df['r']))
#y = df['r'].astype(float)
#plt.plot(x, y)
plt.plot(df["r"])
plt.xlabel('episode')
plt.ylabel('reward')
plt.tight_layout()
plt.show()

"""
# エピソード長のプロット
x = range(len(df['l']))
y = df['l'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('episode len')
plt.show()
"""

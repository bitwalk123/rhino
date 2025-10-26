import os

import matplotlib.pyplot as plt
import numpy as np

from funcs.ios import get_excel_sheet
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    # 推論用データフレーム
    file_excel = "ticks_20250828.xlsx"
    code = "7011"
    path_excel = os.path.join(res.dir_collection, file_excel)
    df = get_excel_sheet(path_excel, code)
    # print(df)
    price_open = df["Price"].iloc[0]
    df["price_ratio"] = (df["Price"] / price_open - 1.0) * 10.
    # df["price_diff"] = np.clip(df["Price"].diff() / 10, -1, 1)
    df["ma_030"] = (df["Price"].rolling(30, min_periods=1).mean() / price_open - 1.0) * 10
    df["ma_060"] = (df["Price"].rolling(60, min_periods=1).mean() / price_open - 1.0) * 10
    df["ma_180"] = (df["Price"].rolling(180, min_periods=1).mean() / price_open - 1.0) * 10
    df["ma_300"] = (df["Price"].rolling(300, min_periods=1).mean() / price_open - 1.0) * 10
    df["ma_diff"] = np.clip((df["ma_060"] - df["ma_300"]) * 10, -1, 1)

    plt.plot(df["price_ratio"])
    # plt.plot(df["ma_030"])
    # plt.plot(df["ma_060"])
    # plt.plot(df["ma_180"])
    plt.plot(df["ma_diff"])
    plt.plot()
    plt.grid()
    plt.show()

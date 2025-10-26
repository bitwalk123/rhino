import os

import matplotlib.pyplot as plt

from funcs.ios import get_excel_sheet
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    # 推論用データフレーム
    file_excel = "ticks_20250819.xlsx"
    code = "7011"
    path_excel = os.path.join(res.dir_collection, file_excel)
    df = get_excel_sheet(path_excel, code)
    print(df)

    plt.plot(df["Price"])
    plt.grid()
    plt.show()
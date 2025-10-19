import pandas as pd
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

from structs.res import AppRes


class Chart(FigureCanvas):
    """
    チャート用 FigureCanvas の雛形
    """

    def __init__(self, res: AppRes):
        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 11
        # ダークモードの設定
        plt.style.use("dark_background")

        # Figure オブジェクト
        self.figure = Figure()
        # 軸領域
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)


class SimpleTrendChart(Chart):
    """
    学習曲線用
    """

    def __init__(self, res: AppRes):
        super().__init__(res)
        # 余白設定
        self.figure.subplots_adjust(left=0.1, right=0.99, top=0.90, bottom=0.15)

    def updateData(self, df: pd.DataFrame, dict_info: dict):
        # トレンドライン（株価）
        ser = df[dict_info["coly"]]
        # 消去
        self.ax.cla()
        # プロット
        self.ax.plot(ser)
        self.ax.set_title(dict_info["title"])
        self.ax.set_xlabel(dict_info["xlabel"])
        self.ax.set_ylabel(dict_info["ylabel"])
        self.ax.grid(True, lw=0.5)
        # 再描画
        self.draw()


class TickChart(Chart):
    """
    ティックチャート用
    """

    def __init__(self, res: AppRes):
        super().__init__(res)
        # 余白設定
        self.figure.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.08)
        # タイムスタンプへ時差を加算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

    def clearPlot(self):
        self.ax.cla()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid()

    def updateData(self, df: pd.DataFrame, title: str):
        # トレンドライン（株価）
        ser = df['Price']
        ser.index = [pd.Timestamp(ts + self.tz, unit='s') for ts in df["Time"]]
        # 消去
        self.ax.cla()
        # プロット
        self.ax.plot(ser, color='lightyellow', linewidth=0.5)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)
        self.ax.set_title(title)
        # 再描画
        self.draw()


class ChartNavigation(NavigationToolbar):
    def __init__(self, chart: FigureCanvas):
        super().__init__(chart)

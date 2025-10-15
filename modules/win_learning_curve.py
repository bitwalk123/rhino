import logging
import os

import pandas as pd

from structs.res import AppRes
from widgets.chart import ChartNavigation, SimpleTrendChart
from widgets.containers import MainWindow
from widgets.statusbar import StatusBar


class WinLearningCurve(MainWindow):
    def __init__(self, res: AppRes, df: pd.DataFrame, file_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = SimpleTrendChart(res)
        dict_info = {
            "title": os.path.basename(file_path),
            "coly": "r",
            "xlabel": "Episode",
            "ylabel": "Reward",
        }
        chart.updateData(df, dict_info)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar()
        statusbar.setSizeGripEnabled(False)
        navbar = ChartNavigation(chart)
        statusbar.addWidget(navbar)
        self.setStatusBar(statusbar)

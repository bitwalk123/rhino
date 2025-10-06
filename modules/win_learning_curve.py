import logging

from funcs.ios import get_excel_sheet
from structs.res import AppRes
from widgets.chart import TickChart, ChartNavigation
from widgets.containers import MainWindow
from widgets.statusbar import StatusBar


class WinLearningCurve(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        #self.chart = chart = TickChart(res)
        #self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        #self.statusbar = statusbar = StatusBar(chart)
        #statusbar.setSizeGripEnabled(False)
        #navbar = ChartNavigation(chart)
        #statusbar.addWidget(navbar)
        #self.setStatusBar(statusbar)


import logging

from funcs.ios import get_excel_sheet
from structs.res import AppRes
from widgets.chart import TickChart, ChartNavigation
from widgets.containers import MainWindow
from widgets.statusbar import StatusBar


class WinTick(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = TickChart(res)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(chart)
        statusbar.setSizeGripEnabled(False)
        navbar = ChartNavigation(chart)
        statusbar.addWidget(navbar)
        self.setStatusBar(statusbar)

    def updateChart(self, path_excel: str, code: str, title: str):
        """
        チャートの更新
        Args:
            path_excel: Excel ファイル（フルパス）
            code: 銘柄コード
            title: チャートのタイトル

        Returns:

        """
        df = get_excel_sheet(path_excel, code)
        self.chart.updateData(df, title)

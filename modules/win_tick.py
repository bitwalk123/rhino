import logging

import pandas as pd
from PySide6.QtCore import Slot, QThread, QObject, Signal

from funcs.ios import get_excel_sheet
from structs.res import AppRes
from widgets.chart import TickChart, ChartNavigation
from widgets.containers import MainWindow
from widgets.statusbar import StatusBar


class TickLoader(QObject):
    finished = Signal(pd.DataFrame)  # 読み込み完了時にデータを送る

    def __init__(self, path_excel: str, code: str):
        super().__init__()
        self.path_excel = path_excel
        self.code = code

    @Slot()
    def run(self):
        df = get_excel_sheet(self.path_excel, self.code)
        self.finished.emit(df)


class WinTick(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res
        self.title: str = ""

        self.thread = None
        self.worker = None

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = TickChart(res)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(res)
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
        self.title = title

        self.thread = thread = QThread()
        self.worker = worker = TickLoader(path_excel, code)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_update_chart)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # スレッドの開始
        thread.start()

    @Slot(pd.DataFrame)
    def on_update_chart(self, df: pd.DataFrame):
        self.chart.updateData(df, self.title)

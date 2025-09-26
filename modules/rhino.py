import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from modules.dock import Dock
from modules.toolbar import ToolBar
from structs.res import AppRes
from widgets.chart import TrendChart
from widgets.statusbar import StatusBar


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.1.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()

        self.setMinimumWidth(1000)
        self.setFixedHeight(400)

        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(res)
        toolbar.clickedPlay.connect(self.on_play)
        self.addToolBar(toolbar)

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = Dock(res)
        dock.listedSheets.connect(self.code_list_updated)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = TrendChart(res)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(chart)
        self.setStatusBar(statusbar)

    def code_list_updated(self, list_code):
        self.toolbar.updateCodeList(list_code)

    def on_play(self):
        list_file = self.dock.getItemsSelected()
        if len(list_file) > 0:
            print(list_file)
        else:
            print("選択されたファイルはありません。")

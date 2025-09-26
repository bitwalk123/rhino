import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from modules.dock import Dock
from modules.toolbar import ToolBar
from structs.res import AppRes


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.1.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()

        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(res)
        self.addToolBar(toolbar)

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = Dock(res)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

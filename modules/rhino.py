import logging

from PySide6.QtCore import Qt

from funcs.ios import get_excel_sheet
from funcs.tse import get_jpx_ticker_list
from modules.dock import Dock
from modules.toolbar import ToolBar
from structs.res import AppRes
from widgets.chart import TickChart
from widgets.containers import MainWindow
from widgets.statusbar import StatusBar


class Rhino(MainWindow):
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
        dock.requestUpdateChart.connect(self.update_chart)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = TickChart(res)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(chart)
        self.setStatusBar(statusbar)

        # ---------------------------------------------------------------------
        # 銘柄コードの保持
        # ---------------------------------------------------------------------
        df = get_jpx_ticker_list(res)
        print(df)

    def code_list_updated(self, list_code):
        self.toolbar.updateCodeList(list_code)

    def on_play(self):
        list_file = self.dock.getItemsSelected()
        if len(list_file) > 0:
            print(list_file)
        else:
            print("選択されたファイルはありません。")

    def update_chart(self, path_excel: str):
        code = self.toolbar.getCurrentCode()
        # print(path_excel, code)
        df = get_excel_sheet(path_excel, code)
        self.chart.updateData(df)

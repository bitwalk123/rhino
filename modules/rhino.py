import logging
import os
import unicodedata

from PySide6.QtCore import Qt

from funcs.commons import get_date_str_from_filename
from funcs.tse import get_jpx_ticker_list
from modules.dock import Dock
from modules.toolbar import ToolBar
from modules.win_tick import WinTick
from structs.res import AppRes
from widgets.containers import MainWindow, TabWidget


class Rhino(MainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.1.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()

        # ---------------------------------------------------------------------
        # 銘柄コードの保持
        # ---------------------------------------------------------------------
        self.dict_name = dict()
        df = get_jpx_ticker_list(res)
        for code, name in zip(df["コード"], df["銘柄名"]):
            self.dict_name[str(code)] = unicodedata.normalize('NFKC', name)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.setMinimumWidth(1000)
        self.setFixedHeight(400)

        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(res)
        toolbar.clickedPlay.connect(self.on_play)
        toolbar.codeChanged.connect(self.update_chart)
        self.addToolBar(toolbar)

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = Dock(res)
        dock.listedSheets.connect(self.code_list_updated)
        dock.selectionChanged.connect(self.file_selection_changed)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # メイン・ウィンドウ
        # ---------------------------------------------------------------------
        base = TabWidget()
        self.setCentralWidget(base)

        self.win_tick = win_tick = WinTick(res)
        base.addTab(win_tick, "ティックチャート")

    def code_list_updated(self, list_code):
        self.toolbar.updateCodeList(list_code)

    def on_play(self):
        """
        学習モデルのトレーニング
        Returns:

        """
        list_file = self.dock.getItemsSelected()
        if len(list_file) > 0:
            print(list_file)
        else:
            print("選択されたファイルはありません。")

    def file_selection_changed(self, path_excel: str):
        pass
        # print(path_excel)

    def update_chart(self, code: str):
        """
        チャートの更新
        Args:
            code: 銘柄コード

        Returns:

        """
        # 現在選択されている Excel ファイル名の取得
        file = self.dock.getCurrentFile()
        # Excel ファイル名から日付情報を取得
        date_str = get_date_str_from_filename(file)
        # チャート・タイトルの文字列生成
        title = f"{self.dict_name[code]}({code}) on {date_str}"
        # Excel ファイルのフルパス
        path_excel = os.path.join(self.res.dir_collection, file)
        # チャートの更新
        self.win_tick.updateChart(path_excel, code, title)

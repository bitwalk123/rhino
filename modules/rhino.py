import logging
import os
import unicodedata

import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent

from funcs.commons import get_date_str_from_filename
from funcs.tse import get_jpx_ticker_list
from modules.agent import PPOAgent
from modules.dock import Dock
from modules.toolbar import ToolBar
from modules.win_learning_curve import WinLearningCurve
from modules.win_tick import WinTick
from structs.res import AppRes
from widgets.containers import MainWindow, TabWidget


class Rhino(MainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.3.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"
    requestTraining = Signal(str, str)
    requestInferring = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()

        # ---------------------------------------------------------------------
        # 銘柄コード、銘柄名の辞書を保持
        # ---------------------------------------------------------------------
        self.dict_name = dict_name = dict()
        df = get_jpx_ticker_list(res)
        for code, name in zip(df["コード"], df["銘柄名"]):
            # 銘柄名は、半角文字にできる文字は変換する
            dict_name[str(code)] = unicodedata.normalize('NFKC', name)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ウィンドウサイズ
        self.setMinimumWidth(1000)
        self.setFixedHeight(400)

        # ウィンドウタイトル
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(res)
        toolbar.clickedPlay.connect(self.on_play)
        toolbar.clickedPig.connect(self.on_pig)
        toolbar.codeChanged.connect(self.update_chart)
        self.addToolBar(toolbar)

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = Dock(res)
        # dock.listedSheets.connect(self.code_list_updated)
        dock.selectionChanged.connect(self.file_selection_changed)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # メイン・ウィンドウ
        # ---------------------------------------------------------------------
        self.tabbase = tabbase = TabWidget()
        self.setCentralWidget(tabbase)
        # タブオブジェクト
        self.win_tick = win_tick = WinTick(res)
        tabbase.addTab(win_tick, "ティックチャート")

        # ---------------------------------------------------------------------
        # スレッド用インスタンス (PPOAgent)
        # ---------------------------------------------------------------------
        # スレッドとワーカー準備
        self.thread = QThread()
        self.worker = PPOAgent(res)
        self.worker.moveToThread(self.thread)

        # GUI → ワーカー
        self.requestTraining.connect(self.worker.train)
        self.requestInferring.connect(self.worker.infer)

        # ワーカー → GUI
        # self.worker.progress.connect(self.on_progress)
        self.worker.finishedTraining.connect(self.on_finished_training)
        self.worker.finishedInferring.connect(self.on_finished_inferring)

        # 終了シグナルでスレッド停止
        # self.worker.finished.connect(self.thread.quit)

        self.thread.start()

    def add_chart_learning_curve(self, df: pd.DataFrame, file_path: str):
        label_tab = "学習曲線"
        for idx in list(reversed(range(self.tabbase.count()))):
            if self.tabbase.tabText(idx) == label_tab:
                self.tabbase.removeTab(idx)
        win_leaning_curve = WinLearningCurve(self.res, df, file_path)
        self.tabbase.addTab(win_leaning_curve, label_tab)
        # win_leaning_curve のタブを表示
        self.tabbase.setCurrentWidget(win_leaning_curve)

    def closeEvent(self, event: QCloseEvent):
        """✕ボタンで安全にスレッド停止"""
        self.logger.info(f"{__name__} MainWindow closing...")
        if self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
        self.logger.info(f"{__name__} Thread safely stopped. Exiting.")
        event.accept()

    def code_list_updated(self, list_code):
        """
        銘柄コードのリストをツールバーのコンボボックスへ反映
        Args:
            list_code: 銘柄コードのリスト（Excel ファイルのシート名）

        Returns:

        """
        self.toolbar.updateCodeList(list_code)

    def file_selection_changed(self, path_excel: str, list_code: list):
        print(f"{path_excel} が選択されました。")
        self.toolbar.updateCodeList(list_code)
        # self.win_tick のタブを表示
        self.tabbase.setCurrentWidget(self.win_tick)

    def get_selected_files(self) -> str:
        # チェックされているファイルをリストで取得
        list_file = self.dock.getItemsSelected()
        if len(list_file) == 0:
            print("選択されたファイルはありません。")
            file = ""
        else:
            file = list_file[0]
        return file

    def on_finished_training(self, file_train: str):
        print("finished training!")
        file_csv = "monitor.csv"
        path_monitor = os.path.join(self.res.dir_log, file_csv)
        if not os.path.exists(path_monitor):
            print(f"{file_csv} is not found in {self.res.dir_log}!")
            return
        df = pd.read_csv(path_monitor, skiprows=[0])
        self.add_chart_learning_curve(df, file_train)

    def on_finished_inferring(self):
        print("finished inferring!")

    def on_pig(self):
        print("DEBUG!")
        file_csv = "training.csv"
        path_monitor = os.path.join(self.res.dir_log, file_csv)
        if not os.path.exists(path_monitor):
            print(f"{file_csv} is not found in {self.res.dir_log}!")
            return
        df = pd.read_csv(path_monitor, skiprows=[0])
        self.add_chart_learning_curve(df)

    def on_play(self):
        """
        学習モデルのトレーニング
        Returns:

        """
        file = self.get_selected_files()
        code = self.toolbar.getCurrentCode()
        self.requestTraining.emit(file, code)

    def update_chart(self, code: str):
        """
        チャートの更新
        Args:
            code: 銘柄コード

        Returns:

        """
        # 現在選択されている Excel ファイル名の取得
        file = self.dock.getCurrentFile()
        if file == "":
            # file が空だったら処理終了
            return

        # Excel ファイル名から日付情報を取得
        date_str = get_date_str_from_filename(file)
        # チャート・タイトルの文字列生成
        title = f"{self.dict_name[code]}({code}) on {date_str}"
        # Excel ファイルのフルパス
        path_excel = os.path.join(self.res.dir_collection, file)
        # チャートの更新
        self.win_tick.updateChart(path_excel, code, title)

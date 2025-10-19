import logging
import os
import unicodedata
from collections import deque

import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QCloseEvent

from funcs.commons import get_date_str_from_filename
from funcs.models import get_ppo_model_path
from funcs.tse import get_jpx_ticker_list
from modules.agent import PPOAgent
from modules.dock import Dock
from modules.toolbar import ToolBar
from modules.win_learning_curve import WinLearningCurve
from modules.win_tick import WinTick
from structs.app_enum import AppMode
from structs.res import AppRes
from widgets.containers import MainWindow, TabWidget
from widgets.labels import LabelStatus
from widgets.statusbar import StatusBar


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
        self.mode = AppMode.TRAIN

        # =====================================================================
        # 銘柄コード、銘柄名の辞書を保持
        # =====================================================================
        self.dict_name = dict_name = dict()
        df = get_jpx_ticker_list(res)
        for code, name in zip(df["コード"], df["銘柄名"]):
            # 銘柄名は、半角文字にできる文字は変換する
            dict_name[str(code)] = unicodedata.normalize('NFKC', name)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ウィンドウサイズ
        self.setMinimumWidth(1200)
        self.setFixedHeight(500)

        # ウィンドウタイトル
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # =====================================================================
        # ツールバー
        # =====================================================================
        self.toolbar = toolbar = ToolBar(res)
        toolbar.clickedPlay.connect(self.on_play)
        toolbar.clickedPig.connect(self.on_pig)
        toolbar.codeChanged.connect(self.update_chart_prep)
        toolbar.modeChanged.connect(self.set_mode)
        self.addToolBar(toolbar)

        # =====================================================================
        # 右側のドック
        # =====================================================================
        self.dock = dock = Dock(res)
        dock.tick_files.selectionChanged.connect(self.file_selection_changed)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # =====================================================================
        # ステータスバー
        # =====================================================================
        self.statusbar = statusbar = StatusBar(res)
        self.msg = msg = LabelStatus()
        self.statusbar.addWidget(msg, stretch=1)
        self.setStatusBar(statusbar)

        # =====================================================================
        # メイン・ウィンドウ
        # =====================================================================
        self.tab_base = tabbase = TabWidget()
        self.tab_base.setTabPosition(TabWidget.TabPosition.South)
        self.setCentralWidget(tabbase)
        # ---------------------------------------------------------------------
        # タブオブジェクト
        # ---------------------------------------------------------------------
        self.win_tick = win_tick = WinTick(res)
        tabbase.addTab(win_tick, "ティックチャート")

        # =====================================================================
        # PPOAgent スレッド用インスタンス
        # =====================================================================
        # ワーカー (PPOAgent) へ渡すファイル名のキュー
        self.deque_file = deque()
        self.code: str = ""  # 銘柄コード
        self.time_sleep: int = 180000
        # ---------------------------------------------------------------------
        # スレッドとワーカー準備
        # ---------------------------------------------------------------------
        self.thread_1 = thread_1 = QThread()
        self.worker_1 = worker_1 = PPOAgent(res)
        worker_1.moveToThread(thread_1)
        # GUI → ワーカー
        self.requestTraining.connect(worker_1.train)
        self.requestInferring.connect(worker_1.infer)
        # ワーカー → GUI
        # worker_1.progress.connect(self.on_progress)
        worker_1.finishedTraining.connect(self.on_finished_training)
        worker_1.finishedInferring.connect(self.on_finished_inferring)
        # 終了シグナルでスレッド停止
        # worker_1.finished.connect(thread_1.quit)
        # スレッド開始
        thread_1.start()

        # =====================================================================
        # ドックにティックファイル一覧を表示
        # =====================================================================
        self.dock.tick_files.setTickFiles()

    def add_chart_learning_curve(self, df: pd.DataFrame, file_path: str):
        label_tab = "学習曲線"
        for idx in list(reversed(range(self.tab_base.count()))):
            if self.tab_base.tabText(idx) == label_tab:
                self.tab_base.removeTab(idx)
        win_leaning_curve = WinLearningCurve(self.res, df, file_path)
        self.tab_base.addTab(win_leaning_curve, label_tab)
        # win_leaning_curve のタブを表示
        self.tab_base.setCurrentWidget(win_leaning_curve)

    def closeEvent(self, event: QCloseEvent):
        """✕ボタンで安全にスレッド停止"""
        self.logger.info(f"{__name__} MainWindow closing...")
        if self.thread_1.isRunning():
            self.worker_1.stop()
            self.thread_1.quit()
            self.thread_1.wait()
        self.logger.info(f"{__name__} Thread safely stopped. Exiting.")
        event.accept()

    def file_selection_changed(self, path_excel: str, list_code: list):
        self.toolbar.updateCodeList(list_code)
        # self.win_tick のタブを表示
        self.tab_base.setCurrentWidget(self.win_tick)

    def get_checked_files(self) -> list:
        # チェックされているファイルをリストで取得
        return self.dock.tick_files.getItemsChecked()

    def inferring(self):
        list_file = self.get_checked_files()
        if len(list_file) == 0:
            print("チェックされたファイルはありません。")
            return
        if len(list_file) > 1:
            print("なぜか複数のティックファイルが選択されています！")
            return

        # ---------------------------------------------------------------------
        # 推論するティックファイル
        # ---------------------------------------------------------------------
        # 推論するファイルを出力
        print("\n***  Inferring  *******")
        file = list_file[0]
        print(file)
        print("***********************")

    def on_finished_training(self, file_train: str):
        # ---------------------------------------------------------------------
        # 報酬の学習曲線をタブに表示
        # ---------------------------------------------------------------------
        file_csv = "monitor.csv"
        path_monitor = os.path.join(self.res.dir_log, file_csv)
        if not os.path.exists(path_monitor):
            print(f"{file_csv} is not found in {self.res.dir_log}!")
            return
        df = pd.read_csv(path_monitor, skiprows=[0])
        self.add_chart_learning_curve(df, file_train)
        # ---------------------------------------------------------------------
        # 次の学習
        # ---------------------------------------------------------------------
        if len(self.deque_file) > 0:
            print("%%% インターバル休憩 %%%")
            QTimer.singleShot(self.time_sleep, self.training)
        else:
            print("%%% finished training(s)! %%%")

    def on_finished_inferring(self):
        print("finished inferring!")

    def on_pig(self):
        print("for DEBUG!")

    def on_play(self):
        """
        学習モデルのトレーニング
        Returns:
        """
        if self.mode == AppMode.TRAIN:
            self.training_prep()
        else:
            self.inferring()

    def set_mode(self, mode: AppMode):
        self.mode = mode
        # ドックのファイル選択モードを切り替える
        self.dock.setSelectionMode(mode)
        # 学習 / 推論モードの切り替え
        if mode == AppMode.TRAIN:
            self.mode_train()
        else:
            self.mode_infer()

    def mode_infer(self):
        pass

    def mode_train(self):
        pass

    def training(self):
        if len(self.deque_file) > 0:
            file = self.deque_file.popleft()
            print(f"%%% start training with {self.code} in {file}. %%%")
            self.requestTraining.emit(file, self.code)
        else:
            print("%%% no tick file for training! %%%")

    def training_prep(self):
        list_file = self.get_checked_files()
        if len(list_file) == 0:
            print("チェックされたファイルはありません。")
            return
        # ---------------------------------------------------------------------
        # 最初の学習
        # ---------------------------------------------------------------------
        # 学習するファイルの一覧を出力
        print("\n***  Training Loop  ***")
        for file in list_file:
            print(file)
        print("***********************")
        # トレーニングするティックデータのファイルリストをキューイング
        self.deque_file = deque(list_file)
        # 銘柄コードの取得
        self.code = code = self.toolbar.getCurrentCode()
        # 新規モデルにするか判定
        if self.toolbar.needNewModel():
            # モデルパスの取得
            model_path, _ = get_ppo_model_path(self.res, code)
            # パスが存在すれば削除
            if os.path.exists(model_path):
                os.remove(model_path)
            # モデルの選択状態をリセット
            self.toolbar.resetNewModelStatus()
        # 学習の開始
        self.training()

    def update_chart(self):
        """
        チャートの更新
        Args:
            code: 銘柄コード

        Returns:
        """
        # 現在選択されている Excel ファイル名の取得
        file = self.dock.tick_files.getCurrentFile()
        code = self.toolbar.getCurrentCode()
        print(f"現在選択されているファイルは {file} です。銘柄コードは {code} です。")
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

    def update_chart_prep(self):
        # GUI をリフレッシュする効果
        QTimer.singleShot(0, self.update_chart)

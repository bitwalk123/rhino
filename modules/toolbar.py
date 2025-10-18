from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar

from funcs.commons import get_icon
from modules.panel_new import PanelNew
from modules.panel_switch import PanelSwitch
from structs.app_enum import AppMode
from structs.res import AppRes
from widgets.combos import ComboBox
from widgets.containers import PadH


class ToolBar(QToolBar):
    clickedPig = Signal()
    clickedPlay = Signal()
    codeChanged = Signal()
    modeChanged = Signal(AppMode)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.code_default = "7011"

        self.setContentsMargins(QMargins(0, 0, 0, 0))

        self.swicth = switch = PanelSwitch()
        switch.selectionChanged.connect(self.changed_mode)
        self.addWidget(switch)

        self.addSeparator()

        self.combo_code = combo_code = ComboBox()
        combo_code.currentIndexChanged.connect(self.changed_code)
        self.addWidget(combo_code)

        action_play = QAction(get_icon(self.res, "play.png"), "開始", self)
        action_play.triggered.connect(self.on_play)
        self.addAction(action_play)

        self.addSeparator()

        self.chk_new = chk_new = PanelNew()
        self.addWidget(chk_new)

        hpad = PadH()
        self.addWidget(hpad)

        # 機能確認やデバッグ用に使用
        action_pig = QAction(get_icon(self.res, "pig.png"), "豚", self)
        action_pig.triggered.connect(self.on_pig)
        self.addAction(action_pig)

    def changed_code(self, idx: int):
        if idx >= 0:
            self.codeChanged.emit()

    def changed_mode(self, mode: AppMode):
        if mode == AppMode.TRAIN:
            self.set_mode_train()
        else:
            self.set_mode_infer()

    def getCurrentCode(self) -> str:
        return self.combo_code.currentText()

    def needNewModel(self) -> bool:
        """
        （古いモデルを消して）新しいモデルを生成するか？
        """
        return self.chk_new.isChecked()

    def on_pig(self):
        self.clickedPig.emit()

    def on_play(self):
        self.clickedPlay.emit()

    def resetNewModelStatus(self):
        """
        「新しいモデルを生成するか？」の
        チェックボックスの選択状態をリセット
        """
        self.chk_new.clearCheckStatus()

    def set_mode_infer(self):
        """
        推論モード
        """
        self.chk_new.setEnabled(False)
        self.modeChanged.emit(AppMode.INFER)

    def set_mode_train(self):
        """
        学習モード
        """
        self.chk_new.setEnabled(True)
        self.modeChanged.emit(AppMode.TRAIN)

    def updateCodeList(self, list_code: list):
        # 一旦、スロットを削除
        self.combo_code.currentIndexChanged.disconnect(self.changed_code)
        # ComboBox のリストをクリア
        self.combo_code.clear()
        # list_code を ComboBox のリストへ反映
        self.combo_code.addItems(list_code)
        # スロットを再定義
        self.combo_code.currentIndexChanged.connect(self.changed_code)
        # デフォルトの銘柄コードがリストにあれば表示
        # ■■■ なかった場合は？ ■■■
        if self.code_default in list_code:
            self.combo_code.setCurrentIndex(list_code.index(self.code_default))

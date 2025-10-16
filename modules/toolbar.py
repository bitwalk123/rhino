from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QToolBar

from funcs.commons import get_icon
from structs.res import AppRes
from widgets.containers import PadH


class ToolBar(QToolBar):
    clickedPig = Signal()
    clickedPlay = Signal()
    codeChanged = Signal(str)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.code_default = "7011"

        self.setContentsMargins(QMargins(0, 0, 0, 0))

        self.combo_code = combo_code = QComboBox()
        combo_code.setContentsMargins(QMargins(0, 0, 0, 0))
        combo_code.currentIndexChanged.connect(self.changed_code)
        self.addWidget(combo_code)

        action_play = QAction(get_icon(self.res, "play.png"), "開始", self)
        action_play.triggered.connect(self.on_play)
        self.addAction(action_play)

        hpad = PadH()
        self.addWidget(hpad)

        action_pig = QAction(get_icon(self.res, "pig.png"), "豚", self)
        action_pig.triggered.connect(self.on_pig)
        self.addAction(action_pig)

    def changed_code(self, idx: int):
        if idx >= 0:
            self.codeChanged.emit(self.getCurrentCode())

    def getCurrentCode(self) -> str:
        return self.combo_code.currentText()

    def on_pig(self):
        self.clickedPig.emit()

    def on_play(self):
        self.clickedPlay.emit()

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

from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QToolBar

from funcs.commons import get_icon
from structs.res import AppRes


class ToolBar(QToolBar):
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

    def changed_code(self, idx: int):
        if idx >= 0:
            self.codeChanged.emit(self.getCurrentCode())

    def getCurrentCode(self) -> str:
        return self.combo_code.currentText()

    def on_play(self):
        self.clickedPlay.emit()

    def updateCodeList(self, list_code: list):
        self.combo_code.clear()
        self.combo_code.addItems(list_code)
        # デフォルトの銘柄コードがリストにあれば表示
        if self.code_default in list_code:
            self.combo_code.setCurrentIndex(list_code.index(self.code_default))

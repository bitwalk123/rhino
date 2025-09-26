from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar, QComboBox

from funcs.commons import get_icon
from structs.res import AppRes


class ToolBar(QToolBar):
    clickedPlay = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.code_default = "7011"

        self.combo_code = combo_code = QComboBox()
        self.addWidget(combo_code)

        action_play = QAction(get_icon(self.res, "play.png"), "開始", self)
        action_play.triggered.connect(self.on_play)
        self.addAction(action_play)

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

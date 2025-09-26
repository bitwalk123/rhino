from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar

from funcs.commons import get_icon
from structs.res import AppRes


class ToolBar(QToolBar):
    clickedPlay = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        action_play = QAction(get_icon(self.res, "play.png"), "開始", self)
        action_play.triggered.connect(self.on_play)
        self.addAction(action_play)

    def on_play(self):
        self.clickedPlay.emit()

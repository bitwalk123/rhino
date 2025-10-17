from PySide6.QtCore import QMargins
from PySide6.QtWidgets import QStatusBar

from structs.res import AppRes


class StatusBar(QStatusBar):
    def __init__(self, res:AppRes):
        super().__init__()
        self.res = res
        self.setContentsMargins(QMargins(0, 0, 0, 0))

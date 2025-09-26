from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.0.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
from PySide6.QtCore import QMargins
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QSizePolicy,
    QTabWidget,
    QWidget,
)


class IndicatorBuySell(QFrame):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(2)
        self.setFixedHeight(5)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        palette = self.palette()
        self.background_default = palette.color(QPalette.ColorRole.Window)
        # print(f"Default background color (RGB): {self.background_default.getRgb()}")

    def setDefault(self):
        self.setStyleSheet("")
        self.setPalette(self.background_default)

    def setBuy(self):
        self.setStyleSheet("QFrame{background-color: magenta;}")

    def setSell(self):
        self.setStyleSheet("QFrame{background-color: cyan;}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class PadH(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)


class PadV(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)


class TabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setStyleSheet("""
            QTabWidget {
                font-family: monospace;
                font-size: 9pt;
            }
        """)


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))

from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel


class LabelRight(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {font-family: monospace;}
        """)
        self.setContentsMargins(QMargins(5, 1, 5, 1))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)


class LabelRightSmall(LabelRight):
    def __init__(self, *args):
        super().__init__(*args)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(7)
        self.setFont(font)

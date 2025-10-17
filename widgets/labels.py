from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLCDNumber,
    QSizePolicy,
)

from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class LabelLeft(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {font-family: monospace;}
        """)
        self.setContentsMargins(QMargins(5, 1, 5, 1))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)


class LabelLeftSmall(LabelLeft):
    def __init__(self, *args):
        super().__init__(*args)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(7)
        self.setFont(font)


class LabelSmall(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setStyleSheet("""
            QLabel {margin: 0 5;}
        """)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(6)
        self.setFont(font)


class LabelStatus(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)


class LCDNumber(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.setDigitCount(12)
        self.display('0.0')


class LCDValueWithTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        # layout
        layout = VBoxLayout()
        self.setLayout(layout)
        # title
        lab_title = LabelSmall(title)
        layout.addWidget(lab_title)
        # LCD
        self.lcd_value = lcd_value = LCDNumber(self)
        layout.addWidget(lcd_value)

    def getValue(self) -> float:
        """
        LCD に表示されている数値を取得
        :return:
        """
        return self.lcd_value.value()

    def setValue(self, value: float):
        """
        LCD に数値を表示
        :param value:
        :return:
        """
        self.lcd_value.display(f"{value:.1f}")

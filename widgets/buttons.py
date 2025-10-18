from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QPushButton,
    QRadioButton,
    QSizePolicy,
)


class RadioButton(QRadioButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QRadioButton {font-family: monospace;}
        """)

class TradeButton(QPushButton):
    def __init__(self, act: str):
        super().__init__()
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(8)
        self.setFont(font)

        if act == "buy":
            self.setText("買　建")
        elif act == "sell":
            self.setText("売　建")
        elif act == "repay":
            self.setText("返　　却")
        else:
            self.setText("不明")

import sys

from PySide6.QtCore import Qt, QMargins
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

class Label(QLabel):
    def __init__(self, string:str):
        super().__init__(string)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.setFrameStyle(QFrame.Shape.WinPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Trade System')

        vbox = QHBoxLayout()
        vbox.setSpacing(2)
        self.setLayout(vbox)

        lab_env = Label("gymnasium.Env")
        lab_env.setStyleSheet("""
            QLabel {
                font-size: 8pt;
                font-family: monospace;
                padding: 1 10 1 10;
                color: #fff;
                background-color: #800;
            }
        """)
        vbox.addWidget(lab_env)

        lab_agent = Label("sb3_contrib.MaskablePPO")
        lab_agent.setStyleSheet("""
            QLabel {
                font-size: 8pt;
                font-family: monospace;
                padding: 1 10 1 10;
                color: #fff;
                background-color: #060;
            }
        """)
        vbox.addWidget(lab_agent)

        lab_policy = Label("MlpPolicy")
        lab_policy.setStyleSheet("""
            QLabel {
                font-size: 8pt;
                font-family: monospace;
                padding: 1 10 1 10;
                color: #fff;
                background-color: #008;
            }
        """)
        vbox.addWidget(lab_policy)

def main():
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
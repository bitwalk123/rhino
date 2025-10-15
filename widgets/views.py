from PySide6.QtCore import QMargins
from PySide6.QtWidgets import QListView


class ListView(QListView):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setStyleSheet("""
            QListView {
                font-family: monospace;
                font-size: 10pt;
            }
        """)

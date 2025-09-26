from PySide6.QtWidgets import QListView


class ListView(QListView):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QListView {
                font-family: monospace;
                font-size: 9pt;
            }
        """)

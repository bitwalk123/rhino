from PySide6.QtCore import QMargins
from PySide6.QtWidgets import QDockWidget

from widgets.containers import Widget, PadH
from widgets.labels import LabelLeftSmall
from widgets.layouts import HBoxLayout


class DockTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        layout = HBoxLayout()
        self.setLayout(layout)

        self.lab_title = LabelLeftSmall(title)
        layout.addWidget(self.lab_title)


class DockWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

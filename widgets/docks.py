from PySide6.QtWidgets import QDockWidget

from widgets.containers import Widget, PadH
from widgets.labels import LabelRightSmall
from widgets.layouts import HBoxLayout


class DockTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        layout = HBoxLayout()
        self.setLayout(layout)

        pad = PadH()
        layout.addWidget(pad)

        self.lab_title = LabelRightSmall(title)
        layout.addWidget(self.lab_title)


class DockWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

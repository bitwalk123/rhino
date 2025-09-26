import logging
import os

from PySide6.QtGui import QStandardItem, QStandardItemModel

from structs.res import AppRes
from widgets.containers import Widget
from widgets.docks import DockWidget, DockTitle
from widgets.layouts import VBoxLayout
from widgets.views import ListView


class Dock(DockWidget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.dock_title = DockTitle("ティックファイル")
        self.setTitleBarWidget(self.dock_title)

        base = Widget()
        self.setWidget(base)

        self.layout = layout = VBoxLayout()
        base.setLayout(layout)

        lv = ListView()
        layout.addWidget(lv)

        model = QStandardItemModel(lv)
        lv.setModel(model)

        files = sorted(os.listdir(res.dir_collection))
        for file in files:
            item = QStandardItem(file)
            item.setCheckable(True)
            model.appendRow(item)

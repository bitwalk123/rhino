import logging
import os

from PySide6.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import QWidget

from funcs.ios import get_excel_sheet_list
from modules.win_tick_files import WinTickFiles
from structs.res import AppRes
from widgets.containers import Widget, TabWidget
from widgets.dockwidgets import DockWidget
from widgets.layouts import VBoxLayout
from widgets.views import ListView


class Dock(DockWidget):
    fileSelectionChanged = Signal(str, list)

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        self.setTitleBarWidget(QWidget(None))

        self.tab_base = tabbase = TabWidget()
        self.tab_base.setTabPosition(TabWidget.TabPosition.East)
        self.setWidget(tabbase)

        # タブオブジェクト
        self.tick_files = tick_files = WinTickFiles(res)
        tick_files.selectionChanged.connect(self.fileSelectionChanged.emit)
        tabbase.addTab(tick_files, "ティックファイル")

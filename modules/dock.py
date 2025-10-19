import logging

from PySide6.QtWidgets import QWidget

from modules.panel_tick_files import PanelTickFiles
from modules.panel_transaction import PanelTransaction
from structs.res import AppRes
from widgets.containers import TabWidget
from widgets.dockwidgets import DockWidget


class Dock(DockWidget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.setFixedWidth(200)
        self.setTitleBarWidget(QWidget(None))

        self.tab_base = tabbase = TabWidget()
        self.tab_base.setTabPosition(TabWidget.TabPosition.East)
        self.setWidget(tabbase)

        # ---------------------------------------------------------------------
        # タブオブジェクト
        # ---------------------------------------------------------------------
        self.tick_files = tick_files = PanelTickFiles(res)
        tabbase.addTab(tick_files, "ティックファイル")
        self.transaction = transaction = PanelTransaction(res)
        tabbase.addTab(transaction, "取引シミュレータ")

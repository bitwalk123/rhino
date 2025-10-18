import logging
import os

from PySide6.QtCore import QModelIndex, QAbstractItemModel, Signal, Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor

from funcs.ios import get_excel_sheet_list
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout
from widgets.views import ListView


class PanelTickFiles(Widget):
    selectionChanged = Signal(str, list)

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        # False: 通常のチェックボックス, True: ラジオボタン風排他的選択
        self.exclusive_check_mode = False

        self.layout = layout = VBoxLayout()
        self.setLayout(layout)

        self.lv = lv = ListView()
        lv.clickedOutsideCheckBox.connect(self.on_clicked)
        layout.addWidget(lv)

        self.model = model = QStandardItemModel(lv)
        model.itemChanged.connect(self.on_item_changed)
        lv.setModel(model)

    def getCurrentFile(self) -> str:
        idx = self.lv.currentIndex().row()
        item: QStandardItem = self.model.item(idx)
        try:
            file: str = item.text()
        except AttributeError:
            file: str = ""

        return file

    def getItemsChecked(self) -> list:
        return [
            item.text()
            for item in (self.model.item(i) for i in range(self.model.rowCount()))
            if item.checkState() == Qt.CheckState.Checked
        ]

    def on_clicked(self, midx: QModelIndex):
        item: QStandardItem = self.model.itemFromIndex(midx)
        file = item.text()
        # print(f"ファイル {file} が選択（クリック）されました。")
        path_excel = os.path.join(self.res.dir_collection, file)
        list_sheet = get_excel_sheet_list(path_excel)
        if len(list_sheet) > 0:
            self.selectionChanged.emit(path_excel, list_sheet)

    def on_item_changed(self, changed_item: QStandardItem):
        if self.exclusive_check_mode and changed_item.checkState() == Qt.CheckState.Checked:
            for i in range(self.model.rowCount()):
                item = self.model.item(i)
                if item is not changed_item:
                    item.setCheckState(Qt.CheckState.Unchecked)

    def setExclusiveCheckMode(self, enabled: bool):
        self.exclusive_check_mode = enabled

    def setTickFiles(self):
        files = sorted(os.listdir(self.res.dir_collection))
        for i, file in enumerate(files):
            item = QStandardItem(file)
            item.setCheckable(True)
            self.model.appendRow(item)

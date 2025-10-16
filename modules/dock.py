import logging
import os

from PySide6.QtCore import (
    QAbstractItemModel,
    QMargins,
    QModelIndex,
    Qt,
    Signal, QItemSelectionModel,
)
from PySide6.QtGui import QStandardItem, QStandardItemModel, QColor

from funcs.ios import get_excel_sheet_list
from structs.res import AppRes
from widgets.containers import Widget
from widgets.docks import DockWidget, DockTitle
from widgets.layouts import VBoxLayout
from widgets.views import ListView


class Dock(DockWidget):
    selectionChanged = Signal(str, list)

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        self.setContentsMargins(QMargins(0, 0, 0, 0))

        self.dock_title = DockTitle("ティックファイル一覧")
        self.setTitleBarWidget(self.dock_title)

        base = Widget()
        self.setWidget(base)
        self.layout = layout = VBoxLayout()
        base.setLayout(layout)

        self.lv = lv = ListView()
        # lv.setMinimumWidth(200)
        # lv.clicked.connect(self.on_clicked)
        lv.clickedOutsideCheckBox.connect(self.on_clicked)
        layout.addWidget(lv)

        self.model = model = QStandardItemModel(lv)
        lv.setModel(model)

    def getCurrentFile(self) -> str:
        idx = self.lv.currentIndex().row()
        print("lv のインデックス", idx)
        # model: QStandardItemModel | QAbstractItemModel = self.lv.model()
        item: QStandardItem = self.model.item(idx)
        try:
            file: str = item.text()
        except AttributeError:
            file: str = ""

        return file

    def getItemsSelected(self) -> list:
        list_file = list()
        model: QStandardItemModel | QAbstractItemModel = self.lv.model()
        for idx in range(model.rowCount()):
            item: QStandardItem = model.item(idx)
            file: str = item.text()
            state: Qt.CheckState = item.checkState()
            if state == Qt.CheckState.Checked:
                list_file.append(file)
        return list_file

    def on_clicked(self, midx: QModelIndex):
        model: QStandardItemModel | QAbstractItemModel = self.lv.model()
        item: QStandardItem = model.itemFromIndex(midx)
        file = item.text()
        print(f"ファイル {file} が選択（クリック）されました。")
        path_excel = os.path.join(self.res.dir_collection, file)
        list_sheet = get_excel_sheet_list(path_excel)
        if len(list_sheet) > 0:
            self.selectionChanged.emit(path_excel, list_sheet)

    def setTickFiles(self):
        files = sorted(os.listdir(self.res.dir_collection))
        for i, file in enumerate(files):
            item = QStandardItem(file)
            item.setCheckable(True)
            # 偶数行と奇数行で色を変える
            if i % 2 == 0:
                item.setBackground(QColor("#f0f0f0"))  # 明るめグレー
            else:
                item.setBackground(QColor("#ffffff"))  # 白
            self.model.appendRow(item)
        """
        midx = self.model.index(0, 0)
        self.lv.selectionModel().select(midx, QItemSelectionModel.SelectionFlag.Select)
        self.lv.setCurrentIndex(midx)  # 表示上も選択状態にする
        self.on_clicked(midx)
        """

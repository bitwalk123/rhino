from PySide6.QtCore import QMargins, Signal, QModelIndex
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QListView, QStyleOptionViewItem, QStyle


class ListView(QListView):
    clickedOutsideCheckBox = Signal(QModelIndex)

    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QListView {font-family: monospace;}
        """)
        self.setContentsMargins(QMargins(0, 0, 0, 0))

    def mousePressEvent(self, event: QMouseEvent):
        mindex = self.indexAt(event.position().toPoint())
        # print("internal-1:", mindex.row())
        if not mindex.isValid():
            return super().mousePressEvent(event)

        rect = self.visualRect(mindex)
        option = QStyleOptionViewItem()
        option.initFrom(self)
        option.rect = rect
        option.state = QStyle.StateFlag.State_Enabled
        option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator

        # チェックボックスの矩形を取得
        style = self.style()
        check_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemCheckIndicator,
            option,
            self
        )

        # チェックボックスがクリックされたか判定
        if check_rect.contains(event.position().toPoint()):
            return super().mousePressEvent(event)
        else:
            print("internal-2:", mindex.row())
            self.clickedOutsideCheckBox.emit(mindex)
            return super().mousePressEvent(event)

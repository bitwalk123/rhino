from PySide6.QtCore import (
    QMargins,
    QModelIndex,
    Signal,
)
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QListView,
    QStyle,
    QStyleOptionViewItem,
)


class ListView(QListView):
    clickedOutsideCheckBox = Signal(QModelIndex)

    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setStyleSheet("""
            QListView {
                font-family: monospace;
                font-size: 10pt;
            }
        """)

    def mousePressEvent(self, event: QMouseEvent):
        index = self.indexAt(event.pos())
        if not index.isValid():
            return super().mousePressEvent(event)

        rect = self.visualRect(index)
        option = QStyleOptionViewItem()
        option.initFrom(self)
        option.rect = rect
        option.state = QStyle.StateFlag.State_Enabled
        option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator

        # チェックボックスの矩形を取得
        style = self.style()
        check_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, self)

        # チェックボックスがクリックされたか判定
        if check_rect.contains(event.pos()):
            return super().mousePressEvent(event)
        else:
            self.clickedOutsideCheckBox.emit(index)
            return super().mousePressEvent(event)
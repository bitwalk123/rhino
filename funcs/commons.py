from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QStyle


def get_builtin_icon(parent: QWidget, name: str) -> QIcon:
    pixmap_icon = getattr(QStyle.StandardPixmap, 'SP_%s' % name)
    return parent.style().standardIcon(pixmap_icon)

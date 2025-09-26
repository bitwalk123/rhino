import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QStyle

from structs.res import AppRes


def get_icon(res: AppRes, img_name: str) -> QIcon:
    return QIcon(os.path.join(res.dir_image, img_name))


def get_builtin_icon(parent: QWidget, name: str) -> QIcon:
    pixmap_icon = getattr(QStyle.StandardPixmap, 'SP_%s' % name)
    return parent.style().standardIcon(pixmap_icon)

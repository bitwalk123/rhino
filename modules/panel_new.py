from PySide6.QtWidgets import QCheckBox

from widgets.containers import Frame
from widgets.layouts import HBoxLayout


class PanelNew(Frame):
    """
    学習・推論の切替用パネル
    """

    def __init__(self):
        super().__init__()
        hbox = HBoxLayout()
        hbox.setSpacing(5)
        self.setLayout(hbox)

        self.chk_new = chk_new = QCheckBox("新規モデル")
        hbox.addWidget(chk_new)

    def isChecked(self) -> bool:
        return self.chk_new.isChecked()

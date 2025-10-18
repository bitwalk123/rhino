from PySide6.QtCore import QMargins
from PySide6.QtWidgets import QRadioButton, QButtonGroup

from widgets.containers import Frame
from widgets.layouts import HBoxLayout


class PanelSwitch(Frame):
    """
    学習・推論の切替用パネル
    """

    def __init__(self):
        super().__init__()
        hbox = HBoxLayout()
        hbox.setSpacing(5)
        self.setLayout(hbox)

        rb_train = QRadioButton("学習")
        hbox.addWidget(rb_train)
        rb_train.toggle()

        rb_infer = QRadioButton("推論")
        hbox.addWidget(rb_infer)

        self.but_group = but_group = QButtonGroup()
        but_group.addButton(rb_train)
        but_group.addButton(rb_infer)
        but_group.buttonToggled.connect(self.selection_changed)

    def selection_changed(self, button, state):
        if state:
            status = 'オン'
        else:
            status = 'オフ'

        print('「%s」を%sにしました。' % (button.text(), status))

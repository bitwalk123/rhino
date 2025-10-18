from PySide6.QtWidgets import QRadioButton, QButtonGroup

from widgets.buttons import RadioButton
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

        rb_train = RadioButton("train")
        hbox.addWidget(rb_train)
        rb_train.toggle()

        rb_infer = RadioButton("infer")
        hbox.addWidget(rb_infer)

        self.but_group = but_group = QButtonGroup()
        but_group.addButton(rb_train)
        but_group.addButton(rb_infer)
        but_group.buttonToggled.connect(self.selection_changed)

    def selection_changed(self, rb: QRadioButton, state: bool):
        if state:
            print(f"「{rb.text()}」をオンにしました。")


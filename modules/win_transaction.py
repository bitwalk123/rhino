import logging

from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class WinTransaction(Widget):

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        self.layout = layout = VBoxLayout()
        self.setLayout(layout)

import logging

from modules.panel import PanelTrading
from structs.res import AppRes
from widgets.containers import Widget
from widgets.labels import LCDValueWithTitle
from widgets.layouts import VBoxLayout


class WinTransaction(Widget):

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        self.layout = layout = VBoxLayout()
        self.setLayout(layout)

        # 現在株価（表示）
        self.price = price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益（表示）
        self.profit = profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益（表示）
        self.total = total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)

        # ---------------------------------------------------------------------
        # 取引用パネル
        # ---------------------------------------------------------------------
        self.trading = trading = PanelTrading()
        # trading.clickedBuy.connect(self.on_buy)
        # trading.clickedSell.connect(self.on_sell)
        # trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(trading)

from PySide6.QtWidgets import QStatusBar

from widgets.chart import TrendChart, ChartNavigation


class StatusBar(QStatusBar):
    def __init__(self, chart: TrendChart):
        super().__init__()
        navbar = ChartNavigation(chart)
        self.addWidget(navbar)
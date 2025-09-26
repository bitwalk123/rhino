from PySide6.QtWidgets import QStatusBar

from widgets.chart import TickChart, ChartNavigation


class StatusBar(QStatusBar):
    def __init__(self, chart: TickChart):
        super().__init__()
        navbar = ChartNavigation(chart)
        self.addWidget(navbar)
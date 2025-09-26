import logging

from PySide6.QtWidgets import QDockWidget

from structs.res import AppRes


class Dock(QDockWidget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

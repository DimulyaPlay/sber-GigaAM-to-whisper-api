from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from app.ui.main_window import MainWindow


def run_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    setTheme(Theme.AUTO)
    window = MainWindow()
    window.show()
    return app.exec()

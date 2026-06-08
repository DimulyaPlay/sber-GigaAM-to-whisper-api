from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from app.backend import RunOptions


class RunWorker(QObject):
    started = Signal()
    log_message = Signal(str)
    progress_changed = Signal(str, int, int, float, float, str)
    file_completed = Signal(object)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, options: RunOptions) -> None:
        super().__init__()
        self._options = options

    @Slot()
    def run(self) -> None:
        from app.backend import AsrService

        self.started.emit()
        service = AsrService()

        def log(message: str) -> None:
            self.log_message.emit(message)

        def progress(stage: str, current: int, total: int, start: float | None, end: float | None, label: str) -> None:
            self.progress_changed.emit(
                stage,
                current,
                total,
                -1.0 if start is None else start,
                -1.0 if end is None else end,
                label,
            )

        try:
            results = service.run(self._options, progress=progress, log=log)
            for item in results:
                self.file_completed.emit(item)
            self.finished.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())

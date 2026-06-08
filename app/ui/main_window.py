from __future__ import annotations

from PySide6.QtCore import QThread
from qfluentwidgets import FluentIcon as FIF, FluentWindow, InfoBar, InfoBarPosition

from app.backend import FileRunResult, RunOptions
from app.ui.pages import LogsPage, OverviewPage, ResultsPage, RunPage
from app.ui.workers import RunWorker


class MainWindow(FluentWindow):
    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: RunWorker | None = None

        self.run_page = RunPage()
        self.run_page.setObjectName("run-page")
        self.results_page = ResultsPage()
        self.results_page.setObjectName("results-page")
        self.logs_page = LogsPage()
        self.logs_page.setObjectName("logs-page")
        self.overview_page = OverviewPage()
        self.overview_page.setObjectName("overview-page")

        self._init_window()
        self._init_navigation()
        self._connect_signals()

    def _init_window(self) -> None:
        self.setWindowTitle("GigaAM ASR")
        self.resize(1280, 860)

    def _init_navigation(self) -> None:
        self.addSubInterface(self.run_page, FIF.PLAY, "Запуск")
        self.addSubInterface(self.results_page, FIF.DOCUMENT, "Результаты")
        self.addSubInterface(self.logs_page, FIF.INFO, "Логи")
        self.addSubInterface(self.overview_page, FIF.APPLICATION, "О приложении")

    def _connect_signals(self) -> None:
        self.run_page.run_requested.connect(self._start_run)

    def _start_run(self, options: RunOptions) -> None:
        if not options.audio_paths:
            self._show_error("Не выбраны аудиофайлы")
            return

        self.run_page.reset_progress()
        self.run_page.set_running(True)
        self.logs_page.append_log("[UI] Starting worker")

        self._thread = QThread(self)
        self._worker = RunWorker(options)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(lambda: self.logs_page.append_log("[WORKER] Started"))
        self._worker.log_message.connect(self.logs_page.append_log)
        self._worker.progress_changed.connect(self.run_page.update_progress)
        self._worker.file_completed.connect(self._handle_file_completed)
        self._worker.finished.connect(self._handle_finished)
        self._worker.failed.connect(self._handle_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(lambda _: self._thread.quit())
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _handle_file_completed(self, result: FileRunResult) -> None:
        self.results_page.add_result(result)

    def _handle_finished(self, results: list[FileRunResult]) -> None:
        self.run_page.set_running(False)
        self.logs_page.append_log(f"[WORKER] Finished: {len(results)} file(s)")
        InfoBar.success(
            title="Готово",
            content=f"Обработано файлов: {len(results)}",
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3500,
            parent=self,
        )

    def _handle_failed(self, traceback_text: str) -> None:
        self.run_page.set_running(False)
        self.logs_page.append_log(traceback_text)
        self._show_error("Ошибка обработки. Трассировка добавлена в логи.")

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _show_error(self, text: str) -> None:
        InfoBar.error(
            title="Ошибка",
            content=text,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000,
            parent=self,
        )

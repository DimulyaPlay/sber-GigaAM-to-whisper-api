from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, CardWidget, ComboBox, LineEdit, PrimaryPushButton, ProgressBar, PushButton, StrongBodyLabel, SwitchButton, TextEdit

from app.backend import FileRunResult, RunOptions


def _make_card(title: str, description: str) -> tuple[CardWidget, QVBoxLayout]:
    card = CardWidget()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)
    layout.addWidget(StrongBodyLabel(title))
    layout.addWidget(CaptionLabel(description))
    return card, layout


class RunPage(QWidget):
    run_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        files_card, files_layout = _make_card("Файлы", "Текущий интерфейс работает с локальными аудиофайлами.")
        file_buttons = QHBoxLayout()
        self.add_files_button = PushButton("Добавить файлы")
        self.clear_files_button = PushButton("Очистить")
        file_buttons.addWidget(self.add_files_button)
        file_buttons.addWidget(self.clear_files_button)
        file_buttons.addStretch(1)
        files_layout.addLayout(file_buttons)

        self.files_list = QListWidget()
        self.files_list.setMinimumHeight(140)
        files_layout.addWidget(self.files_list)
        root.addWidget(files_card)

        params_card, params_layout = _make_card("Параметры", "Пока вынесены только существующие параметры запуска.")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        self.model_combo = ComboBox()
        self.model_combo.addItems(["v3_e2e_rnnt", "v3_rnnt"])
        self.device_combo = ComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.use_cores_edit = LineEdit()
        self.use_cores_edit.setPlaceholderText("Например: 6p или 2l4p")
        self.diarize_switch = SwitchButton()
        self.debug_switch = SwitchButton()
        self.timestamps_switch = SwitchButton()
        self.timestamps_switch.setChecked(True)
        self.num_speakers_edit = LineEdit()
        self.num_speakers_edit.setPlaceholderText("Пусто = авто")
        self.min_speakers_edit = LineEdit()
        self.min_speakers_edit.setPlaceholderText("Пусто")
        self.max_speakers_edit = LineEdit()
        self.max_speakers_edit.setPlaceholderText("Пусто")

        form.addRow("Модель", self.model_combo)
        form.addRow("Устройство", self.device_combo)
        form.addRow("CPU affinity", self.use_cores_edit)
        form.addRow("Диаризация", self.diarize_switch)
        form.addRow("Debug файлы", self.debug_switch)
        form.addRow("Таймкоды", self.timestamps_switch)
        form.addRow("Num speakers", self.num_speakers_edit)
        form.addRow("Min speakers", self.min_speakers_edit)
        form.addRow("Max speakers", self.max_speakers_edit)
        params_layout.addLayout(form)
        root.addWidget(params_card)

        run_card, run_layout = _make_card("Запуск", "Фоновый worker выполняет обработку и передает прогресс сигналами.")
        self.status_label = BodyLabel("Ожидание запуска")
        self.file_progress = ProgressBar()
        self.step_progress = ProgressBar()
        self.file_progress.setRange(0, 100)
        self.step_progress.setRange(0, 100)
        self.start_button = PrimaryPushButton("Запустить")

        run_layout.addWidget(self.status_label)
        run_layout.addWidget(QLabel("Файлы"))
        run_layout.addWidget(self.file_progress)
        run_layout.addWidget(QLabel("Текущий этап"))
        run_layout.addWidget(self.step_progress)
        run_layout.addWidget(self.start_button, alignment=Qt.AlignLeft)
        root.addWidget(run_card)
        root.addStretch(1)

        self.add_files_button.clicked.connect(self._pick_files)
        self.clear_files_button.clicked.connect(self.files_list.clear)
        self.start_button.clicked.connect(self._emit_run_request)

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите аудиофайлы",
            "",
            "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.mp4 *.mkv);;All files (*.*)",
        )
        for file_path in files:
            if not self._has_file(file_path):
                self.files_list.addItem(file_path)

    def _has_file(self, file_path: str) -> bool:
        return any(self.files_list.item(index).text() == file_path for index in range(self.files_list.count()))

    def _parse_optional_int(self, widget: LineEdit) -> int | None:
        text = widget.text().strip()
        return int(text) if text else None

    def _emit_run_request(self) -> None:
        options = RunOptions(
            audio_paths=[self.files_list.item(i).text() for i in range(self.files_list.count())],
            model=self.model_combo.currentText(),
            diarize=self.diarize_switch.isChecked(),
            num_speakers=self._parse_optional_int(self.num_speakers_edit),
            min_speakers=self._parse_optional_int(self.min_speakers_edit),
            max_speakers=self._parse_optional_int(self.max_speakers_edit),
            debug=self.debug_switch.isChecked(),
            no_timestamps=not self.timestamps_switch.isChecked(),
            use_cores=self.use_cores_edit.text().strip() or None,
            device=self.device_combo.currentText(),
        )
        self.run_requested.emit(options)

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.add_files_button.setEnabled(not running)
        self.clear_files_button.setEnabled(not running)

    def update_progress(self, stage: str, current: int, total: int, start: float, end: float, label: str) -> None:
        self.status_label.setText(label)
        if stage == "files":
            self.file_progress.setValue(0 if total <= 0 else int(current * 100 / total))
            return

        self.step_progress.setValue(0 if total <= 0 else int(current * 100 / total))
        if start >= 0 and end >= 0:
            self.status_label.setText(f"{label}: {start:.1f}s - {end:.1f}s")

    def reset_progress(self) -> None:
        self.status_label.setText("Ожидание запуска")
        self.file_progress.setValue(0)
        self.step_progress.setValue(0)


class ResultsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._reports: dict[str, FileRunResult] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        card, layout = _make_card("Результаты", "Список готовых файлов и текстовый отчет по выбранному элементу.")
        content = QGridLayout()
        content.setColumnStretch(0, 1)
        content.setColumnStretch(1, 2)

        self.results_list = QListWidget()
        self.report_view = TextEdit()
        self.report_view.setReadOnly(True)

        content.addWidget(self.results_list, 0, 0)
        content.addWidget(self.report_view, 0, 1)
        layout.addLayout(content)
        root.addWidget(card)

        self.results_list.currentItemChanged.connect(self._show_report)

    def add_result(self, result: FileRunResult) -> None:
        self._reports[result.source_path] = result
        item = QListWidgetItem(f"{Path(result.source_path).name} -> {Path(result.output_path).name}")
        item.setData(Qt.UserRole, result.source_path)
        self.results_list.addItem(item)
        self.results_list.setCurrentItem(item)

    def _show_report(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            self.report_view.clear()
            return
        result = self._reports[current.data(Qt.UserRole)]
        self.report_view.setPlainText(
            "\n".join(
                [
                    f"Source: {result.source_path}",
                    f"Output: {result.output_path}",
                    f"Device: {result.device}",
                    f"Segments: {result.segments_count}",
                    f"ASR time: {result.asr_time_sec:.3f}s",
                    f"Total time: {result.total_time_sec:.3f}s",
                    "",
                    result.report_text,
                ]
            )
        )


class LogsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)
        card, layout = _make_card("Логи", "Сообщения от backend и worker-слоя.")
        self.log_view = TextEdit()
        self.log_view.setReadOnly(True)
        self.clear_button = PushButton("Очистить")
        layout.addWidget(self.log_view)
        layout.addWidget(self.clear_button, alignment=Qt.AlignLeft)
        root.addWidget(card)
        self.clear_button.clicked.connect(self.log_view.clear)

    def append_log(self, message: str) -> None:
        self.log_view.append(message)


class OverviewPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)
        card, layout = _make_card(
            "Архитектура",
            "Базовый GUI-каркас под дальнейшее расширение настроек, источников файлов и экспортов.",
        )
        for text in [
            "Главное окно: Fluent navigation и отдельные страницы.",
            "Worker: фоновый запуск ASR через QObject + QThread.",
            "Связь: сигналы прогресса, логов, ошибок и завершения.",
            "Backend: переиспользуемый сервис, общий для GUI и CLI.",
        ]:
            layout.addWidget(BodyLabel(text))
        root.addWidget(card)
        root.addStretch(1)

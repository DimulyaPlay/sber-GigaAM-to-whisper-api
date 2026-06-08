from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", message=r"(?s).*TensorFloat-32 \(TF32\) has been disabled.*")

ProgressCallback = Callable[[str, int, int, float | None, float | None, str], None]
LogCallback = Callable[[str], None]


def get_runtime_base() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        internal = exe_dir / "_internal"
        return internal if internal.exists() else exe_dir
    return Path(__file__).resolve().parent.parent


BASE = get_runtime_base()
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

FFMPEG_DIR = BASE / "ffmpeg"
if FFMPEG_DIR.exists():
    try:
        os.add_dll_directory(str(FFMPEG_DIR))
    except Exception:
        pass
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

CACHE_DIR = BASE / "models"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")

import gigaam  # noqa: E402
from gigaam.diarization_utils import diarize_audio, pack_speaker_segments, transcribe_diarized_segments  # noqa: E402


@dataclass(slots=True)
class RunOptions:
    audio_paths: list[str]
    model: str = "v3_e2e_rnnt"
    diarize: bool = False
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    debug: bool = False
    no_timestamps: bool = False
    use_cores: str | None = None
    device: str = "auto"


@dataclass(slots=True)
class FileRunResult:
    source_path: str
    output_path: str
    report_text: str
    stats_path: str | None = None
    debug_paths: list[str] = field(default_factory=list)
    audio_duration_sec: float | None = None
    asr_time_sec: float = 0.0
    total_time_sec: float = 0.0
    segments_count: int = 0
    device: str = "cpu"


def detect_cpu_topology() -> tuple[int, int]:
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil

        logical = psutil.cpu_count(logical=True) or logical
        physical = psutil.cpu_count(logical=False) or physical
    except Exception:
        if logical >= 2 and logical % 2 == 0:
            physical = logical // 2
    physical = max(1, min(physical, logical))
    return logical, physical


def parse_use_cores(spec: str) -> tuple[int, int]:
    spec = spec.lower().replace(" ", "")
    parts = re.findall(r"(\d+)([lp])", spec)
    if not parts or "".join(n + t for n, t in parts) != spec:
        raise ValueError("Некорректный формат use_cores. Примеры: 6p, 2l4p, 2p2l")

    logical_count = 0
    physical_count = 0
    for count_str, kind in parts:
        count = int(count_str)
        if kind == "l":
            logical_count += count
        else:
            physical_count += count

    if logical_count == 0 and physical_count == 0:
        raise ValueError("Параметр use_cores не может задавать 0 ядер")
    return logical_count, physical_count


def build_affinity_from_spec(spec: str) -> tuple[list[int] | None, int, list[str]]:
    logical_req, physical_req = parse_use_cores(spec)
    logical_total, physical_total = detect_cpu_topology()
    logs = [
        f"[INFO] CPU topology: logical={logical_total}, physical={physical_total}",
        f"[INFO] Requested cores: {physical_req}p + {logical_req}l",
    ]

    if logical_total == physical_total * 2:
        primary = list(range(0, logical_total, 2))
        sibling = list(range(1, logical_total, 2))
        affinity = sorted(set(primary[:physical_req] + sibling[:logical_req]))
        workers = len(affinity)
        logs.append(f"[INFO] CPU affinity = {affinity}")
        logs.append(f"[INFO] CPU workers = {workers}")
        return affinity, workers, logs

    total_req = logical_req + physical_req
    affinity = list(range(min(total_req, logical_total)))
    workers = len(affinity)
    logs.append("[WARN] Could not split physical/logical cores reliably, using first logical CPUs")
    logs.append(f"[INFO] CPU affinity = {affinity}")
    logs.append(f"[INFO] CPU workers = {workers}")
    return affinity, workers, logs


def apply_cpu_runtime_from_spec(spec: str | None, log: LogCallback) -> int | None:
    if not spec:
        return None

    affinity, workers, logs = build_affinity_from_spec(spec)
    for line in logs:
        log(line)

    os.environ["OMP_NUM_THREADS"] = str(workers)
    os.environ["MKL_NUM_THREADS"] = str(workers)
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"

    if affinity:
        try:
            import psutil

            psutil.Process().cpu_affinity(affinity)
        except Exception as exc:
            log(f"[WARN] Failed to apply cpu affinity: {exc}")
    return workers


def ffprobe_duration_seconds(path: str) -> float | None:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return None


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f} sec"
    minutes, seconds = divmod(sec, 60)
    if minutes < 60:
        return f"{int(minutes):02}:{seconds:05.2f}"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"


def get_process_affinity() -> list[int] | None:
    try:
        import psutil

        return psutil.Process().cpu_affinity()
    except Exception:
        return None


def get_cpu_stats(use_cores: str | None) -> dict:
    logical_total, physical_total = detect_cpu_topology()
    requested_logical = 0
    requested_physical = 0
    affinity = None
    affinity_applied = False

    if use_cores:
        try:
            requested_logical, requested_physical = parse_use_cores(use_cores)
        except Exception:
            pass
        affinity = get_process_affinity()
        affinity_applied = affinity is not None and len(affinity) > 0

    return {
        "logical_total": logical_total,
        "physical_total": physical_total,
        "affinity": affinity,
        "affinity_applied": affinity_applied,
        "requested_logical": requested_logical,
        "requested_physical": requested_physical,
    }


def get_gpu_stats(torch) -> dict:
    if not torch.cuda.is_available():
        return {}

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "name": torch.cuda.get_device_name(idx),
        "index": idx,
        "total_mem_gb": props.total_memory / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(idx) / (1024**3),
        "allocated_gb": torch.cuda.memory_allocated(idx) / (1024**3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(idx) / (1024**3),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(idx) / (1024**3),
    }


def build_run_report(
    *,
    source: Path,
    options: RunOptions,
    device: str,
    duration: float | None,
    utterances_count: int,
    asr_elapsed: float,
    total_elapsed: float,
    cpu_stats: dict,
    gpu_stats: dict | None,
) -> str:
    speed_x = (duration / asr_elapsed) if (duration and asr_elapsed > 0) else None
    lines = [
        f"file: {source.name}",
        f"path: {source}",
        f"model: {options.model}",
        f"mode: {'diarization+asr' if options.diarize else 'longform_asr'}",
        f"device: {device}",
        f"audio_duration_sec: {duration:.2f}" if duration is not None else "audio_duration_sec: unknown",
        f"segments_out: {utterances_count}",
        f"asr_time_sec: {asr_elapsed:.3f}",
        f"total_time_sec: {total_elapsed:.3f}",
        f"asr_time_human: {format_seconds(asr_elapsed)}",
        f"total_time_human: {format_seconds(total_elapsed)}",
        f"speed_x: {speed_x:.2f}" if speed_x is not None else "speed_x: unknown",
        f"platform: {platform.platform()}",
    ]

    if device == "cpu":
        lines.extend(
            [
                "",
                "[cpu]",
                f"logical_total: {cpu_stats['logical_total']}",
                f"physical_total: {cpu_stats['physical_total']}",
            ]
        )
        if cpu_stats.get("affinity_applied"):
            lines.extend(
                [
                    f"process_affinity: {cpu_stats['affinity']}",
                    f"requested_physical: {cpu_stats['requested_physical']}",
                    f"requested_logical: {cpu_stats['requested_logical']}",
                ]
            )

    if device == "cuda" and gpu_stats:
        lines.extend(
            [
                "",
                "[gpu]",
                f"name: {gpu_stats['name']}",
                f"index: {gpu_stats['index']}",
                f"total_mem_gb: {gpu_stats['total_mem_gb']:.2f}",
                f"allocated_gb: {gpu_stats['allocated_gb']:.2f}",
                f"reserved_gb: {gpu_stats['reserved_gb']:.2f}",
                f"peak_allocated_gb: {gpu_stats['peak_allocated_gb']:.2f}",
                f"peak_reserved_gb: {gpu_stats['peak_reserved_gb']:.2f}",
            ]
        )

    return "\n".join(lines)


class AsrService:
    def __init__(self) -> None:
        self._torch = None

    def _import_runtime(self) -> None:
        import gigaam.decoder  # noqa: F401
        import gigaam.decoding  # noqa: F401
        import gigaam.encoder  # noqa: F401
        import pyannote.audio.models  # noqa: F401
        import pyannote.audio.models.embedding  # noqa: F401
        import pyannote.audio.models.segmentation  # noqa: F401
        import torch

        self._torch = torch

    def _resolve_device(self, preference: str, log: LogCallback) -> str:
        assert self._torch is not None
        cuda_available = self._torch.cuda.is_available()
        if preference == "cpu":
            device = "cpu"
        elif preference == "cuda":
            device = "cuda" if cuda_available else "cpu"
            if device == "cpu":
                log("[WARN] CUDA requested but unavailable, using CPU")
        else:
            device = "cuda" if cuda_available else "cpu"
        log(f"[INFO] device = {device}")
        return device

    def run(self, options: RunOptions, progress: ProgressCallback, log: LogCallback) -> list[FileRunResult]:
        if not options.audio_paths:
            raise ValueError("Не выбраны аудиофайлы")

        cores_to_use = apply_cpu_runtime_from_spec(options.use_cores, log)
        self._import_runtime()
        assert self._torch is not None

        if cores_to_use is not None:
            try:
                self._torch.set_num_threads(cores_to_use)
                self._torch.set_num_interop_threads(1)
            except Exception as exc:
                log(f"[WARN] Failed to configure torch threads: {exc}")

        device = self._resolve_device(options.device, log)
        cpu_stats = get_cpu_stats(options.use_cores)
        model = gigaam.load_model(
            options.model,
            download_root=str(CACHE_DIR / "gigaam"),
            device=device,
        )

        results: list[FileRunResult] = []
        total_files = len(options.audio_paths)

        for file_index, audio_path in enumerate(options.audio_paths, start=1):
            source = Path(audio_path)
            progress("files", file_index - 1, total_files, None, None, f"Подготовка {source.name}")
            if not source.exists():
                log(f"[SKIP] File not found: {source}")
                continue

            log(f"[ASR] {source}")
            file_started = time.perf_counter()
            if device == "cuda":
                try:
                    self._torch.cuda.synchronize()
                    self._torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            duration = ffprobe_duration_seconds(str(source))
            if duration is not None:
                log(f"[INFO] Audio duration: {duration:.1f} sec")

            def step_cb(current: int, total: int, bounds: tuple[float, float]) -> None:
                start, end = bounds
                progress("asr", current, total, start, end, f"ASR {source.name}")

            debug_paths: list[str] = []
            asr_started = time.perf_counter()

            if options.diarize:
                diar_segments = diarize_audio(
                    str(source),
                    device=self._torch.device(device),
                    num_speakers=options.num_speakers,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                )
                if device != "cuda":
                    log("[WARN] Diarization on CPU can take significant time")

                if options.debug:
                    debug_raw = source.with_name(f"{source.stem}_diarization_raw.txt")
                    with debug_raw.open("w", encoding="utf-8") as handle:
                        for seg in diar_segments:
                            handle.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}\n")
                    debug_paths.append(str(debug_raw))

                diar_segments = pack_speaker_segments(
                    diar_segments,
                    max_duration=24.0,
                    max_gap=9.0,
                    min_duration=0.35,
                )

                if options.debug:
                    debug_packed = source.with_name(f"{source.stem}_diarization.txt")
                    with debug_packed.open("w", encoding="utf-8") as handle:
                        for seg in diar_segments:
                            handle.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}\n")
                    debug_paths.append(str(debug_packed))

                utterances = transcribe_diarized_segments(model, str(source), diar_segments, progress=step_cb)
            else:
                utterances = model.transcribe_longform(str(source), progress=step_cb)
                if options.debug:
                    debug_asr = source.with_name(f"{source.stem}_asr_raw.txt")
                    with debug_asr.open("w", encoding="utf-8") as handle:
                        for utt in utterances:
                            start, end = utt["boundaries"]
                            handle.write(f"[{start:.2f}-{end:.2f}] {utt['transcription']}\n")
                    debug_paths.append(str(debug_asr))

            if device == "cuda":
                try:
                    self._torch.cuda.synchronize()
                except Exception:
                    pass

            asr_elapsed = time.perf_counter() - asr_started
            log(f"[INFO] Segments: {len(utterances)}")

            output_path = source.with_suffix(".txt")
            with output_path.open("w", encoding="utf-8") as handle:
                total_segments = len(utterances)
                for item_index, utt in enumerate(utterances, start=1):
                    transcription = utt["transcription"]
                    start, end = utt["boundaries"]
                    speaker = utt.get("speaker")
                    prefix = f"[{speaker}] " if speaker else ""
                    if options.no_timestamps:
                        line = f"{prefix}{transcription}"
                    else:
                        line = f"{prefix}[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}"
                    handle.write(line + "\n")
                    progress("write", item_index, total_segments, start, end, f"Запись {source.name}")

            log(f"[OK] -> {output_path}")

            if device == "cuda":
                try:
                    self._torch.cuda.synchronize()
                except Exception:
                    pass

            total_elapsed = time.perf_counter() - file_started
            gpu_stats = get_gpu_stats(self._torch) if device == "cuda" else {}
            report_text = build_run_report(
                source=source,
                options=options,
                device=device,
                duration=duration,
                utterances_count=len(utterances),
                asr_elapsed=asr_elapsed,
                total_elapsed=total_elapsed,
                cpu_stats=cpu_stats,
                gpu_stats=gpu_stats,
            )

            stats_path: str | None = None
            if options.debug:
                stats_file = source.with_name(f"{source.stem}_stats.txt")
                with stats_file.open("w", encoding="utf-8") as handle:
                    handle.write(report_text + "\n")
                stats_path = str(stats_file)
                debug_paths.append(stats_path)
                log(f"[DEBUG] -> {stats_file}")

            results.append(
                FileRunResult(
                    source_path=str(source),
                    output_path=str(output_path),
                    report_text=report_text,
                    stats_path=stats_path,
                    debug_paths=debug_paths,
                    audio_duration_sec=duration,
                    asr_time_sec=asr_elapsed,
                    total_time_sec=total_elapsed,
                    segments_count=len(utterances),
                    device=device,
                )
            )

            progress("files", file_index, total_files, None, None, f"Завершено {source.name}")

        return results

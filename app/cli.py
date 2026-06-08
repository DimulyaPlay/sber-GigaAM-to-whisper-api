from __future__ import annotations

import argparse
import sys

from app.backend import AsrService, RunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GigaAM longform ASR")
    parser.add_argument("audio", nargs="*", help="Пути к аудиофайлам")
    parser.add_argument("--gui", action="store_true", help="Запустить графический интерфейс")
    parser.add_argument("--server", action="store_true", help="Запустить OpenAI-compatible STT HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Host для STT server")
    parser.add_argument("--port", type=int, default=8000, help="Port для STT server")
    parser.add_argument("--server-api-key", default=None, help="Опциональный Bearer API key для локального server")
    parser.add_argument("--model", default="v3_e2e_rnnt", help="Модель GigaAM")
    parser.add_argument("--diarize", action="store_true", help="Включить диаризацию")
    parser.add_argument("--num-speakers", type=int, default=None, help="Точное число спикеров")
    parser.add_argument("--min-speakers", type=int, default=None, help="Минимум спикеров")
    parser.add_argument("--max-speakers", type=int, default=None, help="Максимум спикеров")
    parser.add_argument("--debug", action="store_true", help="Сохранять debug-артефакты")
    parser.add_argument("--no-timestamps", action="store_true", help="Сохранить текст без таймкодов")
    parser.add_argument("--use-cores", type=str, default=None, help="Например: 6p, 2l4p, 2p2l")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", action="store_true", help="Принудительно использовать CPU")
    device_group.add_argument("--gpu", action="store_true", help="Принудительно использовать GPU/CUDA")
    return parser


def run_cli(args: argparse.Namespace) -> int:
    if args.server:
        from app.server import run_server

        return run_server(args)

    if args.gui or not args.audio:
        from app.ui.app import run_gui

        return run_gui()

    device = "auto"
    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "cuda"

    options = RunOptions(
        audio_paths=args.audio,
        model=args.model,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        debug=args.debug,
        no_timestamps=args.no_timestamps,
        use_cores=args.use_cores,
        device=device,
    )

    service = AsrService()

    def log(message: str) -> None:
        print(message)

    def progress(stage: str, current: int, total: int, start: float | None, end: float | None, label: str) -> None:
        if total <= 0:
            return
        if start is not None and end is not None:
            print(f"\r[{stage.upper()}] {current}/{total} [{start:.1f}-{end:.1f}s] {label}", end="", flush=True)
        else:
            print(f"\r[{stage.upper()}] {current}/{total} {label}", end="", flush=True)
        if current >= total:
            print()

    try:
        service.run(options, progress=progress, log=log)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

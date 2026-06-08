from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GigaAM OpenAI-compatible STT server")
    parser.add_argument("--server", action="store_true", default=True, help="Запустить STT HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Host для STT server")
    parser.add_argument("--port", type=int, default=8000, help="Port для STT server")
    parser.add_argument("--server-api-key", default=None, help="Опциональный Bearer API key для локального server")
    parser.add_argument("--model", default="v3_e2e_rnnt", help="Модель GigaAM")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", action="store_true", help="Принудительно использовать CPU")
    device_group.add_argument("--gpu", action="store_true", help="Принудительно использовать GPU/CUDA")
    return parser


def run_cli(args: argparse.Namespace) -> int:
    from app.server import run_server

    return run_server(args)

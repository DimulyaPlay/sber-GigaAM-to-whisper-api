from __future__ import annotations

import argparse
import asyncio
import html
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from aiohttp import web


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


DEFAULT_MODEL = "v3_e2e_rnnt"
MODEL_ALIASES = {
    "gigaam": DEFAULT_MODEL,
    "gigaam-v3": DEFAULT_MODEL,
    "whisper-1": DEFAULT_MODEL,
}
VISIBLE_MODELS = ("gigaam", "whisper-1", DEFAULT_MODEL)


def log(message: str) -> None:
    print(f"[STT] {message}", flush=True)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]
    duration: float | None
    model: str


class GigaAMTranscriber:
    def __init__(self, default_model: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self.default_model = default_model
        self.device_preference = device
        self._torch = None
        self._device: str | None = None
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _import_runtime(self) -> None:
        if self._torch is not None:
            return

        import gigaam.decoder  # noqa: F401
        import gigaam.decoding  # noqa: F401
        import gigaam.encoder  # noqa: F401
        import torch

        self._torch = torch

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device

        self._import_runtime()
        assert self._torch is not None
        if self.device_preference == "cpu":
            self._device = "cpu"
        elif self.device_preference == "cuda":
            self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
        else:
            self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
        log(f"device = {self._device}")
        return self._device

    def _normalize_model_name(self, requested_model: str | None) -> str:
        model_name = (requested_model or self.default_model).strip() or self.default_model
        effective_model = MODEL_ALIASES.get(model_name, self.default_model)
        if effective_model != model_name:
            log(f"requested model {model_name!r} mapped to {effective_model}")
        elif model_name != self.default_model:
            log(f"requested model {model_name!r} ignored, using {self.default_model}")
        return effective_model

    def _load_model(self, model_name: str) -> Any:
        self._import_runtime()
        if model_name in self._models:
            return self._models[model_name]

        import gigaam

        started = time.perf_counter()
        log(f"loading model {model_name} from {CACHE_DIR / 'gigaam'}")
        model = gigaam.load_model(
            model_name,
            download_root=str(CACHE_DIR / "gigaam"),
            device=self._resolve_device(),
        )
        self._models[model_name] = model
        log(f"model {model_name} loaded in {time.perf_counter() - started:.2f}s")
        return model

    def transcribe(self, audio_path: Path, requested_model: str | None) -> TranscriptionResult:
        with self._lock:
            started = time.perf_counter()
            model_name = self._normalize_model_name(requested_model)
            model = self._load_model(model_name)
            duration = ffprobe_duration_seconds(str(audio_path))
            log(f"transcribing {audio_path.name}, model={model_name}, duration={duration}")

            try:
                if duration is not None and duration <= 25.0:
                    text = model.transcribe(str(audio_path)).strip()
                    segments = [{"id": 0, "start": 0.0, "end": duration, "text": text}]
                else:
                    segments = _normalize_segments(model.transcribe_longform(str(audio_path)))
                    text = _join_segment_text(segments)
            except ValueError as exc:
                if "Too long wav file" not in str(exc):
                    raise
                segments = _normalize_segments(model.transcribe_longform(str(audio_path)))
                text = _join_segment_text(segments)

            log(f"transcription done in {time.perf_counter() - started:.2f}s, chars={len(text)}")
            return TranscriptionResult(
                text=text,
                segments=segments,
                duration=duration,
                model=model_name,
            )


def _normalize_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        start, end = segment["boundaries"]
        text = str(segment.get("transcription", "")).strip()
        item: dict[str, Any] = {
            "id": index,
            "start": float(start),
            "end": float(end),
            "text": text,
        }
        if "speaker" in segment:
            item["speaker"] = segment["speaker"]
        segments.append(item)
    return segments


def _join_segment_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(segment["text"] for segment in segments if segment["text"]).strip()


def _format_timestamp(seconds: float, separator: str) -> str:
    millis = round(seconds * 1000)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{ms:03}"


def _format_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp(segment["start"], ",")
        end = _format_timestamp(segment["end"], ",")
        blocks.append(f"{index}\n{start} --> {end}\n{segment['text']}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _format_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        start = _format_timestamp(segment["start"], ".")
        end = _format_timestamp(segment["end"], ".")
        blocks.append(f"{start} --> {end}\n{segment['text']}")
        blocks.append("")
    return "\n".join(blocks)


def _json_error(
    message: str,
    *,
    status: int = 400,
    param: str | None = None,
    code: str | None = None,
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
        status=status,
    )


def _has_valid_auth(request: web.Request, server_api_key: str | None) -> bool:
    if not server_api_key:
        return True

    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {server_api_key}":
        return True
    return request.headers.get("X-API-Key") == server_api_key


def _safe_suffix(filename: str | None, content_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix

    mapping = {
        "audio/flac": ".flac",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
    }
    return mapping.get(content_type or "", ".wav")


async def _read_audio_request(request: web.Request) -> tuple[bytes, str, dict[str, str]]:
    fields: dict[str, str] = dict(request.query)
    content_type = request.content_type or ""

    if request.content_type == "multipart/form-data":
        audio_bytes: bytes | None = None
        suffix = ".wav"
        reader = await request.multipart()
        async for part in reader:
            if part.name == "file":
                audio_bytes = await part.read(decode=False)
                suffix = _safe_suffix(part.filename, part.headers.get("Content-Type"))
            elif part.name:
                fields[part.name] = (await part.text()).strip()
        if audio_bytes is None:
            raise web.HTTPBadRequest(text="Missing required multipart field 'file'")
        return audio_bytes, suffix, fields

    if content_type.startswith("audio/") or content_type == "application/octet-stream":
        audio_bytes = await request.read()
        if not audio_bytes:
            raise web.HTTPBadRequest(text="Request body is empty")
        suffix = _safe_suffix(None, content_type)
        return audio_bytes, suffix, fields

    raise web.HTTPUnsupportedMediaType(
        text="Use multipart/form-data with a 'file' field, or send an audio/* body"
    )


async def create_transcription(request: web.Request) -> web.StreamResponse:
    server_api_key = request.app["server_api_key"]
    if not _has_valid_auth(request, server_api_key):
        log("auth failed")
        return _json_error("Invalid API key", status=401, code="invalid_api_key")

    try:
        audio_bytes, suffix, fields = await _read_audio_request(request)
    except web.HTTPException as exc:
        log(f"bad request: {exc.status} {exc.text or exc.reason}")
        return _json_error(exc.text or exc.reason, status=exc.status)

    requested_model = fields.get("model")
    response_format = fields.get("response_format", "json").lower()
    log(
        "audio request: "
        f"bytes={len(audio_bytes)}, suffix={suffix}, model={requested_model!r}, "
        f"language={fields.get('language')!r}, response_format={response_format}"
    )
    if response_format not in {"json", "text", "verbose_json", "srt", "vtt"}:
        return _json_error(
            "Unsupported response_format. Use json, text, verbose_json, srt, or vtt.",
            param="response_format",
        )

    transcriber: GigaAMTranscriber = request.app["transcriber"]
    with TemporaryDirectory(prefix="gigaam-stt-") as temp_dir:
        audio_path = Path(temp_dir) / f"request{suffix}"
        audio_path.write_bytes(audio_bytes)
        try:
            result = await asyncio.to_thread(transcriber.transcribe, audio_path, requested_model)
        except ValueError as exc:
            log(f"transcription value error: {exc}")
            return _json_error(str(exc), param="model")
        except Exception as exc:
            log(f"transcription failed: {exc}")
            return _json_error(f"Transcription failed: {exc}", status=500)

    if response_format == "text":
        return web.Response(text=result.text, content_type="text/plain")
    if response_format == "srt":
        return web.Response(text=_format_srt(result.segments), content_type="text/plain")
    if response_format == "vtt":
        return web.Response(text=_format_vtt(result.segments), content_type="text/vtt")
    if response_format == "verbose_json":
        return web.json_response(
            {
                "task": "transcribe",
                "language": fields.get("language") or "ru",
                "duration": result.duration,
                "text": result.text,
                "segments": result.segments,
                "model": result.model,
            }
        )
    return web.json_response({"text": result.text})


async def list_models(request: web.Request) -> web.Response:
    models = [
        {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "local-gigaam",
        }
        for model in VISIBLE_MODELS
    ]
    return web.json_response({"object": "list", "data": models})


async def transcription_info(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "message": "Send POST multipart/form-data with file, model, language, response_format.",
            "endpoint": "/v1/audio/transcriptions",
        }
    )


async def health(request: web.Request) -> web.Response:
    transcriber: GigaAMTranscriber = request.app["transcriber"]
    return web.json_response(
        {
            "status": "ok",
            "default_model": transcriber.default_model,
            "device": transcriber._device or transcriber.device_preference,
        }
    )


async def index(request: web.Request) -> web.Response:
    endpoint = f"http://{request.host}/v1/audio/transcriptions"
    text = (
        "<h1>GigaAM STT server</h1>"
        "<p>OpenAI-compatible transcription endpoint:</p>"
        f"<pre>POST {html.escape(endpoint)}</pre>"
    )
    return web.Response(text=text, content_type="text/html")


@web.middleware
async def access_log_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    started = time.perf_counter()
    log(f"--> {request.method} {request.path_qs} content_type={request.content_type}")
    try:
        response = await handler(request)
        elapsed = (time.perf_counter() - started) * 1000
        log(f"<-- {request.method} {request.path} {response.status} {elapsed:.1f}ms")
        return response
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        log(f"<-- {request.method} {request.path} error after {elapsed:.1f}ms: {exc}")
        raise


def create_app(
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    server_api_key: str | None = None,
) -> web.Application:
    app = web.Application(client_max_size=512 * 1024**2, middlewares=[access_log_middleware])
    app["transcriber"] = GigaAMTranscriber(default_model=model, device=device)
    app["server_api_key"] = server_api_key
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/v1/audio/transcriptions", transcription_info)
    app.router.add_post("/v1/audio/transcriptions", create_transcription)
    return app


def run_server(args: argparse.Namespace) -> int:
    device = "auto"
    if getattr(args, "cpu", False):
        device = "cpu"
    elif getattr(args, "gpu", False):
        device = "cuda"

    app = create_app(
        model=args.model,
        device=device,
        server_api_key=args.server_api_key,
    )
    print(f"[STT] listening on http://{args.host}:{args.port}")
    print(f"[STT] endpoint: http://{args.host}:{args.port}/v1/audio/transcriptions")
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0

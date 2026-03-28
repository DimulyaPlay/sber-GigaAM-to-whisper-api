import os
import sys
import warnings
import subprocess
from pathlib import Path
import re
# .venv_cpu/Scripts/pyinstaller.exe --onedir --noconfirm --clean --name GigaAM_ASR_CPU --add-data "ffmpeg;ffmpeg" --add-data "models;models" --add-data ".venv\Lib\site-packages\pyannote\audio\telemetry\config.yaml;pyannote\audio\telemetry" main.py
# .venv/Scripts/pyinstaller.exe --onedir --noconfirm --clean --name GigaAM_ASR_GPU --add-data "ffmpeg;ffmpeg" --add-data "models;models" --add-data ".venv\Lib\site-packages\pyannote\audio\telemetry\config.yaml;pyannote\audio\telemetry" main.py

# 1) Глушим шумный warning ДО импортов, которые тянут pyannote
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torchcodec is not installed correctly.*",
)
def get_runtime_base() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        internal = exe_dir / "_internal"
        return internal if internal.exists() else exe_dir
    return Path(__file__).resolve().parent

BASE = get_runtime_base()
# 3) Чтобы импорты работали одинаково в dev и в exe
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
# 4) Локальный ffmpeg (ffmpeg.exe + dll рядом)
FFMPEG_DIR = BASE / "ffmpeg"
if FFMPEG_DIR.exists():
    try:
        os.add_dll_directory(str(FFMPEG_DIR))
    except Exception:
        pass
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
# 5) Локальный кэш моделей (ты уже скопировал .cache)
CACHE_DIR = BASE / "models"
# huggingface/pyannote
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    """
    '6p'    -> (0, 6)
    '2l4p'  -> (2, 4)
    '2p2l'  -> (2, 2)
    """
    spec = spec.lower().replace(" ", "")
    parts = re.findall(r"(\d+)([lp])", spec)
    if not parts or "".join(n + t for n, t in parts) != spec:
        raise ValueError(
            "Некорректный формат --use-cores. Примеры: 6p, 2l4p, 2p2l"
        )
    logical_count = 0
    physical_count = 0
    for n, kind in parts:
        n = int(n)
        if n < 0:
            raise ValueError("--use-cores не может содержать отрицательные значения")
        if kind == "l":
            logical_count += n
        else:
            physical_count += n
    if logical_count == 0 and physical_count == 0:
        raise ValueError("--use-cores задаёт 0 ядер")

    return logical_count, physical_count


def build_affinity_from_spec(spec: str) -> tuple[list[int] | None, int, list[str]]:
    """
    Возвращает:
      affinity_list | None,
      workers_count,
      log_lines

    ВАЖНО:
    На Windows affinity работает по logical CPU index.
    Поэтому 'physical' здесь — попытка выбрать по одному потоку на физическое ядро.
    Надёжно разделить physical/logical можно только при простой HT/SMT x2 топологии.
    """
    logical_req, physical_req = parse_use_cores(spec)
    logical_total, physical_total = detect_cpu_topology()
    logs = [
        f"[INFO] CPU topology: logical={logical_total}, physical={physical_total}",
        f"[INFO] Requested cores: {physical_req}p + {logical_req}l",
    ]
    # Случай, когда топология похожа на обычный HT x2:
    # primary threads: 0,2,4,...
    # sibling threads: 1,3,5,...
    if logical_total == physical_total * 2:
        primary = list(range(0, logical_total, 2))   # условно "physical"
        sibling = list(range(1, logical_total, 2))   # условно "logical"
        used_physical = primary[:physical_req]
        used_logical = sibling[:logical_req]
        if len(used_physical) < physical_req:
            logs.append(
                f"[WARN] Запрошено {physical_req}p, доступно только {len(primary)} физических шаблонных ядер"
            )
        if len(used_logical) < logical_req:
            logs.append(
                f"[WARN] Запрошено {logical_req}l, доступно только {len(sibling)} логических шаблонных потоков"
            )
        affinity = sorted(set(used_physical + used_logical))
        workers = len(affinity)
        logs.append(f"[INFO] CPU affinity = {affinity}")
        logs.append(f"[INFO] CPU workers = {workers}")
        return affinity, workers, logs
    # fallback: не умеем отделить physical/logical надёжно
    total_req = logical_req + physical_req
    affinity = list(range(min(total_req, logical_total)))
    workers = len(affinity)
    logs.append(
        "[WARN] Не удалось надёжно разделить physical/logical ядра для этой топологии. "
        "Будут использованы первые logical CPU."
    )
    logs.append(f"[INFO] CPU affinity = {affinity}")
    logs.append(f"[INFO] CPU workers = {workers}")
    return affinity, workers, logs


def apply_cpu_runtime_from_spec(spec: str | None) -> int | None:
    """
    Если spec=None:
      ничего не ограничиваем, оставляем поведение системы как есть.
    Если spec задан:
      пытаемся выставить affinity и ограничить число потоков.
    Возвращает workers_count или None.
    """
    if not spec:
        print("[INFO] CPU mode = system default (all available)")
        return None
    affinity, workers, logs = build_affinity_from_spec(spec)
    for line in logs:
        print(line)
    os.environ["OMP_NUM_THREADS"] = str(workers)
    os.environ["MKL_NUM_THREADS"] = str(workers)
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    if affinity:
        try:
            import psutil
            psutil.Process().cpu_affinity(affinity)
        except Exception as e:
            print(f"[WARN] Не удалось применить cpu affinity: {e}")
    return workers


def ffprobe_duration_seconds(path: str) -> float | None:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return None
import gigaam
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GigaAM longform ASR (offline-friendly)")
    parser.add_argument("audio", nargs="+", help="Пути к аудиофайлам (можно несколько)")
    parser.add_argument("--model", default="v3_e2e_rnnt", help="Имя модели GigaAM (v3_rnnt чистый текст, v3_e2e_rnnt нормализованный текст)")
    parser.add_argument(
        "--use-cores",
        type=str,
        default=None,
        help="Ограничить CPU и попытаться привязать процесс к ядрам p-физические, l-логические. Примеры: 6p, 2l4p, 2p2l"
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", action="store_true", help="Принудительно использовать CPU")
    device_group.add_argument("--gpu", action="store_true", help="Принудительно использовать GPU/CUDA")
    args = parser.parse_args()
    try:
        cores_to_use = apply_cpu_runtime_from_spec(args.use_cores)
    except ValueError as e:
        parser.error(str(e))
    # форсим импорты для PyInstaller/Hydra
    import gigaam.encoder
    import gigaam.decoder
    import gigaam.decoding
    import gigaam.onnx_utils
    import pyannote.audio.models
    import pyannote.audio.models.segmentation
    import torch
    if cores_to_use is not None:
        try:
            torch.set_num_threads(cores_to_use)
            torch.set_num_interop_threads(1)
        except Exception as e:
            print(f"[WARN] Не удалось настроить torch threads: {e}")
    cuda_available = torch.cuda.is_available()
    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "cuda"
        if not cuda_available:
            print("Указан --gpu, но CUDA недоступна. Используется CPU")
            device = "cpu"
    else:
        device = "cuda" if cuda_available else "cpu"
    print("[INFO] device =", device)
    model = gigaam.load_model(
        args.model,
        download_root=str(CACHE_DIR / "gigaam"),
        device=device,
    )
    for audio_path in args.audio:
        p = Path(audio_path)
        if not p.exists():
            print(f"[SKIP] Файл не найден: {p}")
            continue
        print(f"[ASR] {p}")
        out_txt = p.with_suffix(".txt")
        dur = ffprobe_duration_seconds(str(p))
        if dur is not None:
            print(f"[INFO] Длительность: {dur:.1f} сек")
        def cb(i, total, b):
            start, end = b
            print(f"\r[ASR] {i}/{total} [{start:.1f}-{end:.1f}s]", end="", flush=True)
        utterances = model.transcribe_longform(str(p), progress=cb)
        print()  # чтобы после \r перейти на новую строку
        print(f"[INFO] Сегментов: {len(utterances)}")
        with out_txt.open("w", encoding="utf-8") as f:
            total = len(utterances)
            for i, utt in enumerate(utterances, 1):
                transcription = utt["transcription"]
                start, end = utt["boundaries"]
                line = f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}"
                f.write(line + "\n")
                # прогресс в консоль
                if dur is not None:
                    pct = min(100.0, (end / dur) * 100.0) if dur > 0 else 0.0
                    print(f"\r[WRITE] {i}/{total}  ~{pct:5.1f}% ({end:6.1f}s/{dur:6.1f}s)", end="", flush=True)
                else:
                    print(f"\r[WRITE] {i}/{total}", end="", flush=True)
        print()
        print(f"[OK] -> {out_txt}")
if __name__ == "__main__":
    main()
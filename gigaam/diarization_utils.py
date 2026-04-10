import os
from typing import Dict, List

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Pipeline

from .preprocess import SAMPLE_RATE, load_audio

_PIPELINE = None


def resolve_local_pipeline_path(repo_id: str) -> str:
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        pass

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            f"Model {repo_id} not found locally and HF_TOKEN is not set."
        )
    return snapshot_download(repo_id=repo_id, token=hf_token)


def get_diarization_pipeline(
    device: torch.device,
    model_id: str = "pyannote/speaker-diarization-community-1",
):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE.to(device)

    local_path = resolve_local_pipeline_path(model_id)
    _PIPELINE = Pipeline.from_pretrained(local_path)
    return _PIPELINE.to(device)


def diarize_audio(
    wav_file: str,
    device: torch.device,
    model_id: str = "pyannote/speaker-diarization-community-1",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> List[Dict]:
    pipeline = get_diarization_pipeline(device=device, model_id=model_id)
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    waveform = load_audio(wav_file, sample_rate=SAMPLE_RATE)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    diarization = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
    annotation = getattr(diarization, "speaker_diarization", diarization)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return segments


def pack_speaker_segments(
    speaker_segments: List[Dict],
    max_duration: float = 24.0,
    max_gap: float = 1.0,
    min_duration: float = 0.35,
) -> List[Dict]:
    if not speaker_segments:
        return []

    speaker_segments = sorted(speaker_segments, key=lambda x: x["start"])
    packed = []
    current = dict(speaker_segments[0])

    for seg in speaker_segments[1:]:
        same_speaker = seg["speaker"] == current["speaker"]
        gap = seg["start"] - current["end"]
        new_duration = seg["end"] - current["start"]
        if same_speaker and gap <= max_gap and new_duration <= max_duration:
            current["end"] = seg["end"]
        else:
            packed.append(current)
            current = dict(seg)
    packed.append(current)

    result = []
    for seg in packed:
        duration = seg["end"] - seg["start"]
        if result and duration < min_duration:
            previous = result[-1]
            if (seg["end"] - previous["start"]) <= max_duration:
                previous["end"] = seg["end"]
                continue
        result.append(dict(seg))
    return result


def transcribe_diarized_segments(
    model,
    wav_file: str,
    speaker_segments: List[Dict],
    sample_rate: int = SAMPLE_RATE,
    min_duration: float = 0.35,
    min_samples: int = 320,
    max_chunk_duration: float = 24.0,
    progress=None,
) -> List[Dict]:
    audio = load_audio(wav_file, sample_rate=sample_rate)
    result = []
    total = len(speaker_segments)

    if progress is True:
        print(f"[DIAR-ASR] segments: {total}")

    for i, seg in enumerate(speaker_segments, 1):
        start = seg["start"]
        end = seg["end"]

        if callable(progress):
            progress(i, total, (start, end))
        elif progress is True:
            print(f"\r[DIAR-ASR] {i}/{total} [{start:.2f}-{end:.2f}]", end="", flush=True)

        if end <= start:
            continue

        chunk = audio[int(start * sample_rate): int(end * sample_rate)]
        if chunk.numel() == 0:
            continue

        duration = end - start
        if duration < min_duration or chunk.numel() < min_samples:
            continue

        if duration <= max_chunk_duration:
            text = model.transcribe_tensor(chunk)
            result.append(
                {
                    "speaker": seg["speaker"],
                    "boundaries": (start, end),
                    "transcription": text,
                }
            )
            continue

        parts = []
        local_segments, local_boundaries = segment_tensor_like_longform(
            chunk,
            sr=sample_rate,
            device=model._device,
            max_duration=22.0,
            min_duration=15.0,
            strict_limit_duration=max_chunk_duration,
            new_chunk_threshold=0.2,
        )
        for sub_chunk, (local_start, local_end) in zip(local_segments, local_boundaries):
            sub_duration = local_end - local_start
            if sub_chunk.numel() < min_samples or sub_duration < min_duration:
                continue
            part_text = model.transcribe_tensor(sub_chunk)
            if part_text:
                parts.append(part_text.strip())

        text = " ".join(parts).strip()
        if not text:
            continue

        result.append(
            {
                "speaker": seg["speaker"],
                "boundaries": (start, end),
                "transcription": text,
            }
        )

    if progress is True:
        print()
    return result


def segment_tensor_like_longform(
    audio: torch.Tensor,
    sr: int,
    device: torch.device,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 24.0,
    new_chunk_threshold: float = 0.2,
):
    from .vad_utils import get_pipeline

    pipeline = get_pipeline(device)
    waveform = audio.unsqueeze(0) if audio.ndim == 1 else audio
    sad_segments = pipeline({"waveform": waveform, "sample_rate": sr})

    segments = []
    boundaries = []
    current_start = 0.0
    current_end = 0.0
    current_duration = 0.0
    total_duration = audio.shape[0] / sr

    def flush(start_t: float, end_t: float, duration_t: float):
        if duration_t <= 0:
            return
        if duration_t > strict_limit_duration:
            parts_count = int(duration_t / strict_limit_duration) + 1
            part_duration = duration_t / parts_count
            start_part = start_t
            for _ in range(parts_count):
                end_part = min(start_part + part_duration, end_t)
                segments.append(audio[int(start_part * sr): int(end_part * sr)])
                boundaries.append((start_part, end_part))
                start_part = end_part
        else:
            segments.append(audio[int(start_t * sr): int(end_t * sr)])
            boundaries.append((start_t, end_t))

    for segment in sad_segments.get_timeline().support():
        start = max(0.0, segment.start)
        end = min(total_duration, segment.end)
        if current_duration > new_chunk_threshold and (
            current_duration + (end - current_end) > max_duration
            or current_duration > min_duration
        ):
            flush(current_start, current_end, current_duration)
            current_start = start
        if current_duration <= new_chunk_threshold:
            current_start = start
        current_end = end
        current_duration = current_end - current_start

    if current_duration > new_chunk_threshold:
        flush(current_start, current_end, current_duration)
    if not segments:
        segments = [audio]
        boundaries = [(0.0, total_duration)]
    return segments, boundaries

from typing import List, Dict, Tuple
import os
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Pipeline
from .preprocess import load_audio, SAMPLE_RATE
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
    diarization = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        **kwargs
    )
    annotation = getattr(diarization, "speaker_diarization", diarization)
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    return segments


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_utterances(
    utterances: List[Dict],
    speaker_segments: List[Dict],
) -> List[Dict]:
    result = []
    for utt in utterances:
        start, end = utt["boundaries"]
        best_speaker = "UNK"
        best_overlap = 0.0
        for seg in speaker_segments:
            ov = overlap(start, end, seg["start"], seg["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = seg["speaker"]
        item = dict(utt)
        item["speaker"] = best_speaker
        result.append(item)

    return result


def merge_speaker_segments(
    speaker_segments: List[Dict],
    max_gap: float = 0.8,
    min_duration: float = 1.0,
) -> List[Dict]:
    if not speaker_segments:
        return []
    speaker_segments = sorted(speaker_segments, key=lambda x: x["start"])
    merged = [dict(speaker_segments[0])]
    for seg in speaker_segments[1:]:
        last = merged[-1]
        same_speaker = seg["speaker"] == last["speaker"]
        gap = seg["start"] - last["end"]
        if same_speaker and gap <= max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))
    # второй проход: слишком короткие куски прилипают к соседям того же спикера
    result = []
    i = 0
    while i < len(merged):
        cur = dict(merged[i])
        cur_dur = cur["end"] - cur["start"]
        if cur_dur < min_duration:
            prev_seg = result[-1] if result else None
            next_seg = merged[i + 1] if i + 1 < len(merged) else None
            if prev_seg and prev_seg["speaker"] == cur["speaker"]:
                prev_seg["end"] = cur["end"]
                i += 1
                continue
            if next_seg and next_seg["speaker"] == cur["speaker"]:
                next_seg = dict(next_seg)
                next_seg["start"] = cur["start"]
                merged[i + 1] = next_seg
                i += 1
                continue
        result.append(cur)
        i += 1
    return result


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
    cur = dict(speaker_segments[0])
    for seg in speaker_segments[1:]:
        same_speaker = seg["speaker"] == cur["speaker"]
        gap = seg["start"] - cur["end"]
        new_duration = seg["end"] - cur["start"]
        if same_speaker and gap <= max_gap and new_duration <= max_duration:
            cur["end"] = seg["end"]
        else:
            packed.append(cur)
            cur = dict(seg)
    packed.append(cur)
    # второй проход: короткие куски прилипают к соседям
    result = []
    for seg in packed:
        dur = seg["end"] - seg["start"]
        if result and dur < min_duration:
            prev = result[-1]
            # лучше всего прилипить к предыдущему, если не выходим за лимит
            if (seg["end"] - prev["start"]) <= max_duration:
                prev["end"] = seg["end"]
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
    progress=None,  # None | True | callable
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

        start_i = int(start * sample_rate)
        end_i = int(end * sample_rate)
        chunk = audio[start_i:end_i]

        if chunk.numel() == 0:
            continue

        dur = end - start
        if dur < min_duration or chunk.numel() < min_samples:
            continue

        # обычный случай
        if dur <= max_chunk_duration:
            text = model.transcribe_tensor(chunk)
            result.append({
                "speaker": seg["speaker"],
                "boundaries": (start, end),
                "transcription": text,
            })
            continue

        # fallback: режем длинный speaker-кусок на подкуски
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
            sub_dur = local_end - local_start
            if sub_chunk.numel() < min_samples or sub_dur < min_duration:
                continue
            part_text = model.transcribe_tensor(sub_chunk)
            if part_text:
                parts.append(part_text.strip())
        text = " ".join(parts).strip()
        if not text:
            continue

        result.append({
            "speaker": seg["speaker"],
            "boundaries": (start, end),
            "transcription": text,
        })
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
    curr_start = 0.0
    curr_end = 0.0
    curr_duration = 0.0
    total_dur = audio.shape[0] / sr

    def _flush(start_t: float, end_t: float, dur_t: float):
        if dur_t <= 0:
            return
        if dur_t > strict_limit_duration:
            max_segments = int(dur_t / strict_limit_duration) + 1
            seg_dur = dur_t / max_segments
            s = start_t
            for _ in range(max_segments):
                e = min(s + seg_dur, end_t)
                segments.append(audio[int(s * sr): int(e * sr)])
                boundaries.append((s, e))
                s = e
        else:
            segments.append(audio[int(start_t * sr): int(end_t * sr)])
            boundaries.append((start_t, end_t))

    for segment in sad_segments.get_timeline().support():
        start = max(0.0, segment.start)
        end = min(total_dur, segment.end)
        if curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _flush(curr_start, curr_end, curr_duration)
            curr_start = start
        if curr_duration <= new_chunk_threshold:
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start
    if curr_duration > new_chunk_threshold:
        _flush(curr_start, curr_end, curr_duration)
    if not segments:
        segments = [audio]
        boundaries = [(0.0, total_dur)]
    return segments, boundaries
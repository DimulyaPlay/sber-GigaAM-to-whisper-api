from typing import Dict, List, Tuple, Union

import hydra
import omegaconf
import torch
from torch import Tensor, nn

from .preprocess import SAMPLE_RATE, load_audio

LONGFORM_THRESHOLD = 25 * SAMPLE_RATE


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass through the preprocessor and encoder.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    @torch.inference_mode()
    def transcribe(self, wav_file: str) -> str:
        """
        Transcribes a short audio file into text.
        """
        wav, length = self.prepare_wav(wav_file)
        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        return self.decoding.decode(self.head, encoded, encoded_len)[0]

    @torch.inference_mode()
    def transcribe_tensor(self, wav: Tensor) -> str:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        length = torch.full([1], wav.shape[-1], device=self._device)
        wav = wav.to(self._device).to(self._dtype)

        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long audio chunk, split it before transcribe_tensor().")

        encoded, encoded_len = self.forward(wav, length)
        return self.decoding.decode(self.head, encoded, encoded_len)[0]

    @torch.inference_mode()
    def transcribe_longform(
            self,
            wav_file: str,
            progress=None,  # None | True | callable
            chunk_duration: float = 24.0,
            **kwargs
    ) -> List[Dict[str, Union[str, Tuple[float, float]]]]:
        """
        Transcribes a long audio file by splitting it into fixed-size chunks.

        progress:
          - None: без прогресса
          - True: печатать прогресс в stdout
          - callable(i, total, boundaries): вызывать коллбек
        """
        transcribed_segments = []
        audio = load_audio(wav_file, sample_rate=SAMPLE_RATE)
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        if chunk_samples <= 0:
            raise ValueError("chunk_duration must be positive")

        segments = []
        boundaries = []
        total_samples = int(audio.shape[0])
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            segment = audio[start_sample:end_sample]
            if segment.numel() == 0:
                continue
            segments.append(segment)
            boundaries.append((start_sample / SAMPLE_RATE, end_sample / SAMPLE_RATE))

        total = len(segments)
        if progress is True:
            print(f"[ASR] segments: {total}")

        for i, (segment, segment_boundaries) in enumerate(zip(segments, boundaries), 1):
            if callable(progress):
                progress(i, total, segment_boundaries)
            elif progress is True:
                start, end = segment_boundaries
                print(f"\r[ASR] {i}/{total}  [{start:.2f}-{end:.2f}]", end="", flush=True)

            wav = segment.to(self._device).unsqueeze(0).to(self._dtype)
            length = torch.full([1], wav.shape[-1], device=self._device)
            encoded, encoded_len = self.forward(wav, length)
            result = self.decoding.decode(self.head, encoded, encoded_len)[0]
            transcribed_segments.append(
                {"transcription": result, "boundaries": segment_boundaries}
            )

        if progress is True:
            print()  # чтобы после \r перейти на новую строку

        return transcribed_segments

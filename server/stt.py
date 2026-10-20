"""Speech to text with faster-whisper, running on the laptop."""

from __future__ import annotations

import re
import time

import numpy as np

import config
from audio_utils import pcm_to_float

# Whisper sometimes "hears" these in silence or noise (subtitle credits from its
# training data). Nobody says them to a speaker, so a transcript that is only one
# of them counts as nothing heard. Same list as the Worker.
HALLUCINATIONS = {
    "", "you", "um", "uh", "hmm", "mm", "the end",
    "thanks for watching", "thank you for watching", "thanks for watching bye",
    "please subscribe", "subscribe", "like and subscribe",
    "subtitles by the amara org community", "transcribed by", "subtitles by",
}


def clean_transcript(text: str) -> str:
    """Trim, and treat empty or known-hallucination transcripts as nothing heard."""
    t = re.sub(r"\s+", " ", text or "").strip()
    key = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    if key in HALLUCINATIONS or len(key) < 2:
        return ""
    return t[: config.MAX_INPUT_CHARS]


class STT:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel  # imported here so --help stays fast

        device, compute = config.WHISPER_DEVICE, config.WHISPER_COMPUTE
        try:
            print(f"[stt] loading Whisper '{config.WHISPER_MODEL}' ({device}/{compute})...")
            self._model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute)
        except Exception as exc:  # usually missing CUDA/cuDNN DLLs on Windows
            if device == "cpu":
                raise
            print(f"[stt] {device} failed ({exc}); using the CPU instead")
            self._model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[stt] ready")

    def warmup(self) -> None:
        """Run a second of hiss through Whisper, with and without the voice-activity filter, so the first real
        question isn't the one that pays for waking the model up (the first call after start-up is slower).
        Whatever it "hears" is thrown away, and a failure here never stops the server starting."""
        t0 = time.time()
        hiss = (np.random.default_rng(0).standard_normal(config.SAMPLE_RATE) * 800).astype(np.int16)
        try:
            for vad in (False, True):
                segments, _info = self._model.transcribe(pcm_to_float(hiss), language=config.WHISPER_LANGUAGE,
                                                         beam_size=1, vad_filter=vad,
                                                         condition_on_previous_text=False)
                for _ in segments:                 # the segments are made lazily: this is what runs the model
                    pass
        except Exception as exc:
            print(f"[stt] warm-up failed ({type(exc).__name__}: {exc}); the first question may be slower")
            return
        print(f"[stt] warmed up in {int((time.time() - t0) * 1000)} ms")

    def transcribe(self, pcm: np.ndarray) -> str:
        """pcm: int16 mono at config.SAMPLE_RATE. Returns the raw text it heard."""
        segments, _info = self._model.transcribe(
            pcm_to_float(pcm),
            language=config.WHISPER_LANGUAGE,
            beam_size=1,
            vad_filter=True,                   # drops the silence a noisy mic leaves behind
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,  # stops one bad guess poisoning the next
            initial_prompt=config.WHISPER_HINT or None,
        )
        return " ".join(s.text.strip() for s in segments).strip()

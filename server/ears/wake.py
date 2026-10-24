"""The wake word. openWakeWord scores every 80 ms frame from 0 to 1 for ONE phrase: "hey jarvis" (built in)
or your own model, e.g. "hey friday". Other speech, music and noise stay near 0; crossing WAKE_THRESHOLD
means the phrase was said.

The model files (about 5 MB) are downloaded from openWakeWord's GitHub releases the first time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]           # the server folder: custom model paths are relative to it


class WakeWord:
    def __init__(self, model: str = "hey_jarvis") -> None:
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise RuntimeError("openwakeword isn't installed: pip install -r requirements.txt") from exc
        path = Path(model)
        if path.suffix in (".onnx", ".tflite"):
            path = path if path.is_absolute() else HERE / path
            if not path.exists():
                raise RuntimeError(f"wake word model not found: {path}")
            download_models([path.stem])                 # the shared feature models (an official name is fetched too)
            spec, self.phrase = str(path), path.stem
        else:
            download_models([model])                     # the shared feature models + this phrase, first run only
            spec, self.phrase = model, model
        self.model = Model(wakeword_models=[spec], inference_framework="onnx")
        self.key = next(iter(self.model.models))
        self.phrase = self.phrase.replace("_", " ").split(" v0")[0]

    def score(self, frame: np.ndarray) -> float:
        return float(self.model.predict(frame)[self.key])

    def reset(self) -> None:
        """Forget the last second of audio, so one "hey jarvis" can't trigger twice."""
        self.model.reset()

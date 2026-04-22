from pathlib import Path

import librosa

from config import config
from models.track import Track


def load_audio(path: str, sr: int = 44100) -> tuple:
    """Load raw audio. Returns (y, sr) tuple for backward compatibility."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def load_track(path: str) -> Track:
    """Load an audio file and return a Track object."""
    p = Path(path)
    y, sr = librosa.load(str(p), sr=config.sample_rate, mono=True)
    return Track(
        path=p,
        title=p.stem,
        audio=y,
        sample_rate=sr,
        duration_seconds=len(y) / sr,
    )

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from config import config

logger = logging.getLogger("dj.audio_io.exporter")


def export_mix(
    audio: np.ndarray,
    path: str,
    sr: int | None = None,
    fmt: str | None = None,
) -> None:
    """Export audio array to a file."""
    if sr is None:
        sr = config.sample_rate
    if fmt is None:
        fmt = config.output_format

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(output_path), np.asarray(audio, dtype=np.float32), sr, format=fmt)
    logger.info("Exported mix to %s (%s, %d Hz)", output_path, fmt, sr)

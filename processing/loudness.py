import logging

import numpy as np
import pyloudnorm as pyln

from config import config
from models.track import Track

logger = logging.getLogger("dj.processing.loudness")


def measure_loudness(track: Track) -> Track:
    """Measure integrated loudness in LUFS and set track.loudness_lufs."""
    if track.audio is None or len(track.audio) == 0:
        track.loudness_lufs = None
        logger.warning("Skipping loudness measure for %s (empty audio)", track.title)
        return track

    meter = pyln.Meter(track.sample_rate)
    try:
        track.loudness_lufs = float(meter.integrated_loudness(track.audio))
    except ValueError as exc:
        # pyloudnorm can fail on extremely short arrays.
        track.loudness_lufs = None
        logger.warning("Skipping loudness measure for %s (%s)", track.title, exc)
        return track

    logger.info("Loudness: %.1f LUFS for %s", track.loudness_lufs, track.title)
    return track


def normalize_loudness(track: Track, target_lufs: float | None = None) -> Track:
    """Normalize track audio to target LUFS level."""
    if track.audio is None or len(track.audio) == 0:
        logger.warning("Skipping loudness normalization for %s (empty audio)", track.title)
        return track

    if target_lufs is None:
        target_lufs = config.target_lufs

    if track.loudness_lufs is None:
        measure_loudness(track)
    if track.loudness_lufs is None:
        return track

    try:
        normalized = pyln.normalize.loudness(
            track.audio, track.loudness_lufs, target_lufs
        )
    except ValueError as exc:
        logger.warning("Skipping loudness normalization for %s (%s)", track.title, exc)
        return track

    normalized = np.asarray(normalized, dtype=np.float32)

    # Apply the same gain to stems so they stay in sync with the
    # normalized audio.  Without this, stems are louder/quieter than
    # track.audio, causing volume mismatches during stem-based mixing.
    if track.stems is not None:
        old_rms = np.sqrt(np.mean(track.audio ** 2)) + 1e-10
        new_rms = np.sqrt(np.mean(normalized ** 2)) + 1e-10
        gain = new_rms / old_rms
        if track.stems.vocals is not None:
            track.stems.vocals = (track.stems.vocals * gain).astype(np.float32)
        if track.stems.drums is not None:
            track.stems.drums = (track.stems.drums * gain).astype(np.float32)
        if track.stems.bass is not None:
            track.stems.bass = (track.stems.bass * gain).astype(np.float32)
        if track.stems.other is not None:
            track.stems.other = (track.stems.other * gain).astype(np.float32)

    track.audio = normalized
    track.loudness_lufs = target_lufs
    logger.info("Normalized to %.1f LUFS for %s", target_lufs, track.title)
    return track

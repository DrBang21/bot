"""Beat alignment with double/half time awareness."""

import logging

from models.track import Track
from mixing.bpm_utils import effective_bpm_ratio
from processing.timestretch import time_stretch

logger = logging.getLogger("dj.mixing.alignment")


def align_tracks(track_a: Track, track_b: Track) -> tuple[int, int]:
    """Compute sample offsets to align track_b's first beat with track_a's last downbeat.

    Uses double/half time aware BPM matching — if track_b is at half or
    double tempo of track_a, it stretches to the appropriate multiple.
    Returns (offset_a, offset_b) in samples.
    """
    if track_a.beats is None or track_b.beats is None:
        raise ValueError("Both tracks must have beat data before alignment")
    if track_a.bpm is None or track_b.bpm is None:
        raise ValueError("Both tracks must have BPM data before alignment")
    if track_a.bpm <= 0 or track_b.bpm <= 0:
        raise ValueError("Track BPM values must be > 0 before alignment")

    # Find the best BPM ratio considering 1:1, 2:1, 1:2
    ratio, target_bpm = effective_bpm_ratio(track_a.bpm, track_b.bpm)

    if abs(ratio - 1.0) > 0.03:
        logger.info("Stretching track_b from %.1f to %.1f BPM (ratio: %.3f)",
                     track_b.bpm, target_bpm, ratio)
        time_stretch(track_b, target_bpm)

    # Use downbeats if available, otherwise fall back to regular beats
    a_beats = track_a.downbeats if track_a.downbeats is not None else track_a.beats
    b_beats = track_b.downbeats if track_b.downbeats is not None else track_b.beats

    # Align: start crossfade at track_a's last downbeat, track_b's first beat
    offset_a = int(a_beats[-1] * track_a.sample_rate) if len(a_beats) > 0 else 0
    offset_b = int(b_beats[0] * track_b.sample_rate) if len(b_beats) > 0 else 0

    logger.info("Alignment offsets: a=%d, b=%d samples", offset_a, offset_b)
    return offset_a, offset_b

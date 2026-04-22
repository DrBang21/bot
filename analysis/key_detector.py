import logging

import librosa
import numpy as np

from models.track import Track

logger = logging.getLogger("dj.analysis.key")

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]


def _correlate_key(chroma_vector: np.ndarray) -> str:
    """Match a chroma vector against major/minor key profiles."""
    best_corr = -1.0
    best_key = "C"

    for shift in range(12):
        major_shifted = np.roll(MAJOR_PROFILE, shift)
        minor_shifted = np.roll(MINOR_PROFILE, shift)

        maj_corr = np.corrcoef(chroma_vector, major_shifted)[0, 1]
        min_corr = np.corrcoef(chroma_vector, minor_shifted)[0, 1]
        if not np.isfinite(maj_corr):
            maj_corr = -1.0
        if not np.isfinite(min_corr):
            min_corr = -1.0

        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = PITCH_CLASSES[shift]
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = PITCH_CLASSES[shift] + "m"

    return best_key


def detect_key(track: Track) -> Track:
    """Estimate musical key using chroma features and Krumhansl-Schmuckler."""
    if track.audio is None or len(track.audio) == 0:
        raise ValueError("Track audio must be loaded before key detection")

    chroma = librosa.feature.chroma_cqt(y=track.audio, sr=track.sample_rate)
    chroma_avg = np.mean(chroma, axis=1)

    track.key = _correlate_key(chroma_avg)
    logger.info("Detected key: %s for %s", track.key, track.title)
    return track

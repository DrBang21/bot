import logging

import librosa
import numpy as np

from models.track import Track

logger = logging.getLogger("dj.processing.timestretch")


def time_stretch(track: Track, target_bpm: float) -> Track:
    """Time-stretch track audio to match target BPM without changing pitch."""
    if track.bpm is None:
        raise ValueError("Track BPM must be detected before time stretching")
    if track.bpm <= 0 or target_bpm <= 0:
        raise ValueError("Track BPM and target BPM must be > 0")

    rate = target_bpm / track.bpm
    if abs(rate - 1.0) < 0.001:
        return track  # Already at target BPM

    track.audio = librosa.effects.time_stretch(track.audio, rate=rate)

    # Update metadata to reflect the stretch
    if track.beats is not None:
        track.beats = track.beats / rate
    if track.downbeats is not None:
        track.downbeats = track.downbeats / rate
    if track.sections is not None:
        for section in track.sections:
            section.start /= rate
            section.end /= rate
    if track._bpm_phase is not None:
        track._bpm_phase /= rate
    if track._first_beat is not None:
        track._first_beat /= rate
    if track.stems is not None:
        for name in ("vocals", "drums", "bass", "other"):
            stem = getattr(track.stems, name)
            if stem is not None:
                setattr(track.stems, name, librosa.effects.time_stretch(stem, rate=rate))
    track.duration_seconds = track.num_samples / track.sample_rate
    track.bpm = target_bpm

    logger.info("Time-stretched to %.1f BPM for %s", target_bpm, track.title)
    return track


def time_stretch_audio(audio: np.ndarray, source_bpm: float,
                       target_bpm: float) -> np.ndarray:
    """Time-stretch a raw audio array from source_bpm to target_bpm.

    Unlike time_stretch(), this works on a plain numpy array and doesn't
    modify any Track metadata. Useful for stretching individual stems.
    """
    if source_bpm <= 0 or target_bpm <= 0:
        raise ValueError("source_bpm and target_bpm must be > 0")
    rate = target_bpm / source_bpm
    if abs(rate - 1.0) < 0.001:
        return audio
    result = librosa.effects.time_stretch(audio, rate=rate)
    logger.info("Stretched audio from %.1f to %.1f BPM (rate=%.3f)",
                source_bpm, target_bpm, rate)
    return result.astype(np.float32)

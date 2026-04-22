import logging

import librosa

from models.track import Track

logger = logging.getLogger("dj.processing.pitchshift")

# Semitone-to-key shift mapping for updating track.key
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]


def pitch_shift(track: Track, semitones: float) -> Track:
    """Shift pitch by the given number of semitones without changing tempo."""
    if abs(semitones) < 0.01:
        return track

    track.audio = librosa.effects.pitch_shift(
        track.audio, sr=track.sample_rate, n_steps=semitones
    )

    # Update key if it was detected and shift is an integer
    if track.key is not None and semitones == int(semitones):
        is_minor = track.key.endswith("m")
        root = track.key.rstrip("m")
        if root in PITCH_CLASSES:
            idx = (PITCH_CLASSES.index(root) + int(semitones)) % 12
            track.key = PITCH_CLASSES[idx] + ("m" if is_minor else "")

    logger.info("Pitch-shifted %+.1f semitones for %s", semitones, track.title)
    return track

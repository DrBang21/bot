import logging

from models.track import Track
from mixing.bpm_utils import bpm_difference_pct

logger = logging.getLogger("dj.playlist")

# Camelot wheel compatibility: maps key to Camelot code
CAMELOT_WHEEL = {
    "C": "8B",  "Am": "8A",
    "G": "9B",  "Em": "9A",
    "D": "10B", "Bm": "10A",
    "A": "11B", "F#m": "11A",
    "E": "12B", "C#m": "12A",
    "B": "1B",  "G#m": "1A",
    "F#": "2B", "D#m": "2A",
    "C#": "3B", "A#m": "3A",
    "G#": "4B", "Fm": "4A",
    "D#": "5B", "Cm": "5A",
    "A#": "6B", "Gm": "6A",
    "F": "7B",  "Dm": "7A",
}


def _camelot_distance(key_a: str, key_b: str) -> int:
    """Compute distance on the Camelot wheel (0 = perfect, lower = better)."""
    code_a = CAMELOT_WHEEL.get(key_a)
    code_b = CAMELOT_WHEEL.get(key_b)
    if code_a is None or code_b is None:
        return 12  # Unknown key, max penalty

    num_a, letter_a = int(code_a[:-1]), code_a[-1]
    num_b, letter_b = int(code_b[:-1]), code_b[-1]

    # Distance on the number ring (1-12, wrapping)
    num_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))

    # Penalty for crossing inner/outer ring (A <-> B)
    ring_penalty = 0 if letter_a == letter_b else 1

    return num_dist + ring_penalty


def _bpm_score(bpm_a: float, bpm_b: float) -> float:
    """Score BPM compatibility (0.0 = identical, higher = worse)."""
    try:
        return bpm_difference_pct(bpm_a, bpm_b)
    except ValueError:
        return 1.0


def select_next(current: Track, candidates: list[Track]) -> Track | None:
    """Select the best next track based on BPM and key compatibility."""
    if not candidates:
        return None

    best_score = float("inf")
    best_track = candidates[0]

    for candidate in candidates:
        score = 0.0

        # BPM scoring (weight: 1.0)
        if current.bpm and candidate.bpm:
            score += _bpm_score(current.bpm, candidate.bpm) * 10.0

        # Key scoring (weight: 2.0)
        if current.key and candidate.key:
            score += _camelot_distance(current.key, candidate.key) * 2.0

        if score < best_score:
            best_score = score
            best_track = candidate

    logger.info(
        "Selected next: %s (score: %.2f)", best_track.title, best_score
    )
    return best_track

"""BPM compatibility utilities with double/half time awareness."""


def effective_bpm_ratio(bpm_a: float, bpm_b: float) -> tuple[float, float]:
    """Return the best BPM ratio considering 1:1, 2:1, and 1:2 relationships.

    Returns (best_ratio, effective_bpm_b) where:
      - best_ratio is closest to 1.0 among the three candidates
      - effective_bpm_b is the BPM track_b would be stretched to
    """
    if bpm_a <= 0 or bpm_b <= 0:
        raise ValueError("BPM values must be > 0")

    candidates = [
        (bpm_b / bpm_a, bpm_a),               # 1:1 — stretch B to A
        ((bpm_b * 2) / bpm_a, bpm_a),          # B is half-time of A
        ((bpm_b / 2) / bpm_a, bpm_a),          # B is double-time of A
    ]
    best_ratio, target = min(candidates, key=lambda c: abs(c[0] - 1.0))
    return best_ratio, target


def bpm_difference_pct(bpm_a: float, bpm_b: float) -> float:
    """Percentage difference using the best ratio (double/half time aware).

    Returns a value like 0.03 for 3% difference.
    """
    ratio, _ = effective_bpm_ratio(bpm_a, bpm_b)
    return abs(ratio - 1.0)


def bars_to_samples(bpm: float, bars: int, sample_rate: int) -> int:
    """Convert a number of bars (4/4 time) to samples at the given BPM."""
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    beats_per_bar = 4
    seconds_per_beat = 60.0 / bpm
    seconds = bars * beats_per_bar * seconds_per_beat
    return int(seconds * sample_rate)


def bars_to_seconds(bpm: float, bars: int) -> float:
    """Convert a number of bars (4/4 time) to seconds at the given BPM."""
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    return bars * 4 * (60.0 / bpm)

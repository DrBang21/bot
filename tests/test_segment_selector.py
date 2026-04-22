from pathlib import Path

import numpy as np

from mixing.segment_selector import find_entry_point, find_exit_point, prepare_segments
from models.track import Section, Track


def _track_with_sections(sections: list[Section], duration: float = 120.0) -> Track:
    sr = 44100
    audio = np.zeros(int(duration * sr), dtype=np.float32)
    return Track(
        path=Path("dummy.wav"),
        title="dummy",
        audio=audio,
        sample_rate=sr,
        duration_seconds=duration,
        sections=sections,
    )


def test_find_exit_prefers_last_chorus_or_drop_before_tail() -> None:
    track = _track_with_sections(
        [
            Section("intro", 0.0, 16.0, 0.20),
            Section("chorus", 16.0, 40.0, 0.85),
            Section("verse", 40.0, 64.0, 0.60),
            Section("drop", 64.0, 84.0, 0.90),
            Section("outro", 84.0, 120.0, 0.10),
        ],
        duration=120.0,
    )
    assert find_exit_point(track) == 84.0


def test_find_exit_fallback_uses_latest_strong_non_outro_section() -> None:
    track = _track_with_sections(
        [
            Section("intro", 0.0, 18.0, 0.10),
            Section("verse", 18.0, 44.0, 0.80),
            Section("bridge", 44.0, 72.0, 0.50),
            Section("verse", 72.0, 86.0, 0.75),
            Section("outro", 86.0, 120.0, 0.15),
        ],
        duration=120.0,
    )
    # No chorus/drop labels exist, so fallback should still pick a sensible
    # high-energy exit near the end of the musical body.
    assert find_exit_point(track) == 86.0


def test_find_entry_fallback_uses_highest_energy_when_no_chorus_drop() -> None:
    track = _track_with_sections(
        [
            Section("intro", 0.0, 20.0, 0.20),
            Section("verse", 20.0, 48.0, 0.45),
            Section("verse", 48.0, 64.0, 0.88),
            Section("bridge", 64.0, 80.0, 0.35),
        ],
        duration=120.0,
    )
    entry, section = find_entry_point(track)
    assert section is not None
    assert section.start == 48.0
    assert entry == 32.0


def test_prepare_segments_extends_into_first_post_exit_section() -> None:
    track_a = _track_with_sections(
        [
            Section("intro", 0.0, 12.0, 0.20),
            Section("chorus", 12.0, 36.0, 0.90),
            Section("verse", 36.0, 60.0, 0.55),
            Section("drop", 60.0, 78.0, 0.92),
            Section("verse", 78.0, 96.0, 0.60),  # first section after exit
            Section("outro", 96.0, 120.0, 0.15),
        ],
        duration=120.0,
    )
    track_b = _track_with_sections(
        [
            Section("intro", 0.0, 20.0, 0.20),
            Section("chorus", 20.0, 40.0, 0.80),
        ],
        duration=100.0,
    )

    seg_a, _, _ = prepare_segments(track_a, track_b, is_first=True)
    # Exit point is at 78s (end of drop), but segment should include the first
    # post-exit section through 96s for instrumental-first layering.
    assert abs((seg_a.duration_seconds or 0.0) - 96.0) < 1e-6

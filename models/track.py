from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from analysis.lyric_detector import Lyrics

import numpy as np


@dataclass
class Section:
    """A labelled time region within a track."""
    label: str          # "intro", "verse", "chorus", "drop", "bridge", "outro"
    start: float        # seconds
    end: float          # seconds
    energy: float = 0.0 # average RMS energy (0.0–1.0, normalised within track)


@dataclass
class Stems:
    """Separated audio stems (all at the same sample rate as the parent track)."""
    vocals: Optional[np.ndarray] = field(default=None, repr=False)
    drums: Optional[np.ndarray] = field(default=None, repr=False)
    bass: Optional[np.ndarray] = field(default=None, repr=False)
    other: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def instrumental(self) -> Optional[np.ndarray]:
        """Everything except vocals (drums + bass + other)."""
        parts = [s for s in (self.drums, self.bass, self.other) if s is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return np.sum(np.stack(parts, axis=0), axis=0, dtype=np.float32)


@dataclass
class Track:
    path: Path
    title: str = ""

    # Raw audio (set by loader)
    audio: Optional[np.ndarray] = field(default=None, repr=False)
    sample_rate: int = 44100

    # Analysis results (set by analysis modules)
    bpm: Optional[float] = None
    beats: Optional[np.ndarray] = field(default=None, repr=False)
    key: Optional[str] = None
    downbeats: Optional[np.ndarray] = field(default=None, repr=False)

    # Stem separation (set by stem separator)
    stems: Optional[Stems] = field(default=None, repr=False)

    # Song structure (set by structure analyzer)
    sections: Optional[list[Section]] = field(default=None, repr=False)

    # Lyrics (set by lyric detector)
    lyrics: Optional[Lyrics] = field(default=None, repr=False)

    # Processing state
    loudness_lufs: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Pre-merge sections (finer resolution, used by lyric refinement)
    _pre_merge_sections: Optional[list[Section]] = field(default=None, repr=False)

    # BPM phase offset (seconds) — the time of the first beat in the
    # onset-optimal grid.  Set by detect_bpm's onset-correlation method.
    _bpm_phase: Optional[float] = field(default=None, repr=False)

    # First beat position (seconds) — where the music actually starts
    # rhythmically.  Used for aligning B's audio in transitions.
    _first_beat: Optional[float] = field(default=None, repr=False)

    @property
    def num_samples(self) -> int:
        return len(self.audio) if self.audio is not None else 0

    def get_sections_by_label(self, label: str) -> list[Section]:
        """Return all sections matching the given label."""
        if self.sections is None:
            return []
        return [s for s in self.sections if s.label == label]

    def get_best_section(self, prefer: list[str] | None = None) -> Section | None:
        """Return the highest-energy section, optionally preferring certain labels.

        If *prefer* is given (e.g. ["chorus", "drop"]), only those labels are
        considered.  Falls back to highest-energy across all sections.
        """
        if self.sections is None or len(self.sections) == 0:
            return None
        candidates = self.sections
        if prefer:
            filtered = [s for s in candidates if s.label in prefer]
            if filtered:
                candidates = filtered
        return max(candidates, key=lambda s: s.energy)

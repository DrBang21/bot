"""Quick structure/segment verification utility.

Usage:
    .venv\\Scripts\\python.exe tools\\verify_segments.py
    .venv\\Scripts\\python.exe tools\\verify_segments.py "audio\\a.wav" "audio\\b.wav"
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running from repository root as a standalone script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_io.loader import load_track
from analysis.bpm_detector import detect_bpm
from analysis.beat_tracker import detect_beats
from analysis.structure_analyzer import detect_structure
from mixing.segment_selector import find_entry_point, find_exit_point


def _print_track_summary(path: str) -> tuple:
    track = load_track(path)
    detect_bpm(track)
    detect_beats(track)
    detect_structure(track)

    print(f"\n=== {track.title} ===")
    print(
        f"Duration={track.duration_seconds:.2f}s  BPM={track.bpm:.2f}  "
        f"Beats={len(track.beats) if track.beats is not None else 0}  "
        f"Downbeats={len(track.downbeats) if track.downbeats is not None else 0}"
    )
    if track.sections:
        for s in track.sections:
            print(
                f"{s.label:7s} {s.start:7.2f} -> {s.end:7.2f}  "
                f"dur={s.end - s.start:6.2f}  E={s.energy:.3f}"
            )
    else:
        print("No sections detected.")
    return track


def main() -> int:
    args = sys.argv[1:]
    if not args:
        audio_dir = Path("audio")
        args = sorted(str(p) for p in audio_dir.glob("*.wav") if "dj_mix" not in p.name)

    if len(args) < 1:
        print("No audio files found.")
        return 1

    tracks = [_print_track_summary(a) for a in args]

    if len(tracks) >= 2:
        a, b = tracks[0], tracks[1]
        exit_a = find_exit_point(a)
        entry_b, target = find_entry_point(b)
        target_label = target.label if target else "none"
        target_start = target.start if target else 0.0
        print("\n=== Transition Diagnostics (first pair) ===")
        print(f"A exit:  {exit_a:.2f}s")
        print(f"B entry: {entry_b:.2f}s (target {target_label} @ {target_start:.2f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

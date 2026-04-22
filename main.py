import sys
from pathlib import Path

import numpy as np

from analysis import analyze
from audio_io.exporter import export_mix
from audio_io.loader import load_track
from config import config
from knowledge.vector_store import TransitionKnowledgeBase
from knowledge.seed_knowledge import seed
from mixing.segment_selector import prepare_segments
from mixing.transition_decider import get_eligible_transitions
from mixing.transition_engine import create_transition, set_knowledge_base
from mixing.rag_decider import record_feedback
from utils.logger import logger


def main() -> None:
    # Initialize knowledge base and seed rules
    kb = TransitionKnowledgeBase()
    seed(kb)
    set_knowledge_base(kb)

    audio_dir = Path(config.audio_dir)
    dir_a = audio_dir / "a"
    dir_b = audio_dir / "b"

    # Ensure subdirectories exist
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    patterns = ("*.wav", "*.mp3")

    def find_track(folder: Path) -> str | None:
        for pattern in patterns:
            matches = sorted(folder.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    file_a = find_track(dir_a)
    file_b = find_track(dir_b)

    if file_a is None:
        logger.error("No audio file found in '%s/' — drop track A there", dir_a)
        sys.exit(1)
    if file_b is None:
        logger.error("No audio file found in '%s/' — drop track B there", dir_b)
        sys.exit(1)

    logger.info("Found 2 tracks")

    # Load and analyze both tracks in explicit A → B order
    tracks = []
    for f in (file_a, file_b):
        logger.info("Loading %s...", f)
        track = load_track(f)
        analyze(track)
        tracks.append(track)

        logger.info("  BPM=%.1f  Key=%s  Duration=%.1fs",
                     track.bpm or 0, track.key or "?", track.duration_seconds or 0)

        if track.sections:
            section_str = " | ".join(
                f"{s.label}({s.start:.0f}-{s.end:.0f}s)"
                for s in track.sections
            )
            logger.info("  Structure: %s", section_str)

        if track.stems:
            stems_present = [
                name for name, arr in [
                    ("vocals", track.stems.vocals),
                    ("drums", track.stems.drums),
                    ("bass", track.stems.bass),
                    ("other", track.stems.other),
                ] if arr is not None
            ]
            logger.info("  Stems: %s", ", ".join(stems_present))

        if track.lyrics and hasattr(track.lyrics, 'full_text') and track.lyrics.full_text:
            text = track.lyrics.full_text
            preview = text[:100] + "..." if len(text) > 100 else text
            logger.info("  Lyrics: \"%s\"", preview)

    # Order is explicit: A first, B second
    playlist = tracks
    logger.info("Playlist order: %s", [t.title for t in playlist])

    # Build the mix using segments (best parts only, not whole songs)
    # Each transition returns the COMPLETE audio for that pair:
    #   [segment_a audio] → [transition blend] → [segment_b audio]
    # So we chain them: first transition output, then for subsequent
    # transitions we skip the part of B that was already included.

    if len(playlist) < 2:
        logger.error("Need at least 2 tracks in playlist")
        sys.exit(1)

    output_audio = np.array([], dtype=np.float32)
    strategies_used: list[str] = []

    for i in range(len(playlist) - 1):
        a, b = playlist[i], playlist[i + 1]

        # Pick the best segments from each track
        seg_a, seg_b, lyric_cue = prepare_segments(a, b, is_first=(i == 0))

        logger.info("Segments: %s (%.1fs) -> %s (%.1fs)",
                     a.title, seg_a.duration_seconds or 0,
                     b.title, seg_b.duration_seconds or 0)

        # Show eligible transitions for these segments
        options = get_eligible_transitions(seg_a, seg_b)
        logger.info("Transition %s -> %s:", a.title, b.title)
        for opt in options:
            logger.info("  [%s] score=%.2f %s",
                         ">>>" if opt == options[0] else "   ",
                         opt.score, opt.strategy.name)

        # Execute the best transition — returns complete [A...blend...B] audio
        transition = create_transition(seg_a, seg_b, lyric_cue=lyric_cue)
        strategies_used.append(options[0].strategy.name if options else "unknown")

        if len(output_audio) == 0:
            output_audio = transition
        else:
            output_audio = np.concatenate([output_audio, transition])

    # Export final mix
    output_path = audio_dir / f"dj_mix.{config.output_format}"
    export_mix(output_audio, str(output_path))

    mix_duration = len(output_audio) / config.sample_rate
    logger.info("Done! Mix saved to %s (%.1fs / %.1fmin)",
                output_path, mix_duration, mix_duration / 60)

    # --- Feedback loop ---
    # Ask the user to rate the transition so RAG improves over time.
    try:
        print("\n--- Rate the transition (improves future mixes) ---")
        rating_str = input("Rating 1-5 (or Enter to skip): ").strip()
        if rating_str and rating_str in "12345":
            rating = int(rating_str)
            notes = input("Notes (optional): ").strip()
            # Record feedback for each transition pair
            for i in range(len(playlist) - 1):
                a, b = playlist[i], playlist[i + 1]
                record_feedback(kb, a, b, strategies_used[i], rating, notes)
            print(f"Feedback saved — {kb.count()} entries in knowledge base.")
        else:
            print("Skipped.")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()

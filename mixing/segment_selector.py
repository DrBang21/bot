"""Segment selector: decides WHERE to exit track A and WHERE to enter track B.

Philosophy:
- Track A (first in set) starts from the beginning — a DJ plays from the top.
- The algorithm picks an EXIT point on A: after a high-energy section ends,
  on a natural phrase boundary (downbeat), before the energy drops off.
- The algorithm picks an ENTRY point on B: just before a chorus/drop so the
  transition LANDS on an energetic moment.
"""

import copy
import logging

import numpy as np

from models.track import Section, Track

logger = logging.getLogger("dj.mixing.segments")

# How many seconds of "runway" to give before B's entry section
# so the transition has room to blend in
ENTRY_RUNWAY_SEC = 16.0

# Preferred labels for where to enter track B
ENTRY_LABELS = ["drop", "chorus"]


def find_exit_point(track: Track) -> float:
    """Find the best time (in seconds) to begin exiting track A.

    Scores every chorus/drop candidate on:
      - Position: penalise exiting before 40% of the track (too early)
      - Transition potential: bonus if a bridge or low-energy section follows
      - Progression: slight preference for later choruses so the listener
        hears more of the song before transitioning
    """
    duration = track.duration_seconds or (len(track.audio) / track.sample_rate)

    if not track.sections:
        return duration * 0.7

    energies = [s.energy for s in track.sections]
    if not energies:
        return duration * 0.7

    # Collect every chorus/drop that isn't in the outro tail
    candidates: list[tuple[Section, float]] = []
    for idx, s in enumerate(track.sections):
        if s.label not in ("chorus", "drop"):
            continue
        if s.end >= duration * 0.92:
            continue  # too close to the end

        rel_pos = s.end / duration  # 0-1, where the exit would land

        # --- Position score: penalise very early exits ---
        if rel_pos < 0.30:
            pos_score = 0.2
        elif rel_pos < 0.45:
            pos_score = 0.5
        else:
            pos_score = 0.8 + 0.2 * min(1.0, (rel_pos - 0.45) / 0.35)

        # --- Transition-zone bonus: bridge or low-energy section after? ---
        next_sections = track.sections[idx + 1:]
        tz_bonus = 0.0
        for ns in next_sections:
            if ns.start < s.end:
                continue
            if ns.label == "bridge":
                tz_bonus = 0.25  # best case — bridge after chorus
            elif ns.energy < s.energy * 0.85:
                tz_bonus = 0.10  # low-energy section is decent too
            break  # only check the immediate next section

        score = pos_score + tz_bonus
        candidates.append((s, score))
        logger.debug("Exit candidate: %s @ %.1fs (pos=%.0f%%, score=%.2f, tz=%.2f)",
                      s.label, s.end, rel_pos * 100, score, tz_bonus)

    # If the best chorus/drop candidate exits before 45% of the track,
    # also consider later verse sections so the listener hears more of
    # the song.  Verses score lower than choruses but a late verse beats
    # an early chorus.
    best_chorus_score = max((s for _, s in candidates), default=0.0)
    best_chorus_pos = 0.0
    if candidates:
        best_c, _ = max(candidates, key=lambda x: x[1])
        best_chorus_pos = best_c.end / duration

    if best_chorus_pos < 0.45 or best_chorus_score < 0.70:
        energy_threshold = np.percentile(energies, 50)
        for idx, s in enumerate(track.sections):
            if s.label in ("chorus", "drop", "intro", "outro"):
                continue
            if s.end >= duration * 0.92 or s.end < duration * 0.45:
                continue
            if s.energy < energy_threshold:
                continue
            rel_pos = s.end / duration
            pos_score = 0.6 + 0.3 * min(1.0, (rel_pos - 0.45) / 0.35)
            # Verses get a penalty vs choruses but position can compensate
            verse_penalty = -0.15
            # Bonus if a low-energy section follows (good transition zone)
            tz_bonus = 0.0
            next_sections = track.sections[idx + 1:]
            for ns in next_sections:
                if ns.start < s.end:
                    continue
                if ns.label == "bridge":
                    tz_bonus = 0.20
                elif ns.energy < s.energy * 0.85:
                    tz_bonus = 0.08
                break
            score = pos_score + verse_penalty + tz_bonus
            candidates.append((s, score))
            logger.debug("Exit candidate (verse): %s @ %.1fs (pos=%.0f%%, score=%.2f)",
                          s.label, s.end, rel_pos * 100, score)

    if candidates:
        best_section, best_score = max(candidates, key=lambda x: x[1])
        exit_time = best_section.end
        logger.info("Exit point for %s: %.1fs (after %s, E=%.2f, score=%.2f)",
                     track.title, exit_time, best_section.label,
                     best_section.energy, best_score)
        return exit_time

    # Fallback: latest strong section before the outro
    energy_threshold = np.percentile(energies, 60)
    fallback_section = None
    for s in track.sections:
        if s.end >= duration * 0.90:
            continue
        if s.label == "outro":
            continue
        if s.energy >= energy_threshold:
            fallback_section = s

    if fallback_section is not None:
        exit_time = fallback_section.end
        logger.info("Exit point for %s: %.1fs (after strong %s, E=%.2f)",
                    track.title, exit_time, fallback_section.label, fallback_section.energy)
        return exit_time

    return duration * 0.7


def find_entry_point(track: Track) -> tuple[float, Section | None]:
    """Find the best time (in seconds) to start playing track B.

    Strategy: find the first chorus or drop in the track. The entry point
    is placed BEFORE this section (with runway) so the transition can
    blend into the energetic moment.

    Returns (entry_time_sec, target_section).
    """
    duration = track.duration_seconds or (len(track.audio) / track.sample_rate)

    if not track.sections:
        # No structure — enter from the beginning
        return 0.0, None

    # Find the first chorus or drop
    for s in track.sections:
        if s.label in ENTRY_LABELS:
            # Place entry before this section with runway
            entry_time = max(0.0, s.start - ENTRY_RUNWAY_SEC)
            logger.info("Entry point for %s: %.1fs (runway to %s @ %.1fs, E=%.2f)",
                         track.title, entry_time, s.label, s.start, s.energy)
            return entry_time, s

    # No chorus/drop found — find the highest-energy section
    best = max(track.sections, key=lambda s: s.energy)
    entry_time = max(0.0, best.start - ENTRY_RUNWAY_SEC)
    logger.info("Entry point for %s: %.1fs (runway to %s @ %.1fs, E=%.2f)",
                 track.title, entry_time, best.label, best.start, best.energy)
    return entry_time, best


def extract_segment(track: Track, start_sec: float, end_sec: float) -> Track:
    """Create a trimmed copy of a Track with audio sliced to [start, end].

    Adjusts beats, downbeats, sections, and stems to the new time range.
    The original Track is not modified.
    """
    sr = track.sample_rate
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(len(track.audio), int(end_sec * sr))

    trimmed = copy.copy(track)
    trimmed.audio = track.audio[start_sample:end_sample].copy()
    trimmed.duration_seconds = len(trimmed.audio) / sr

    # Adjust beat arrays to new time base
    if track.beats is not None:
        mask = (track.beats >= start_sec) & (track.beats < end_sec)
        trimmed.beats = track.beats[mask] - start_sec

    if track.downbeats is not None:
        mask = (track.downbeats >= start_sec) & (track.downbeats < end_sec)
        trimmed.downbeats = track.downbeats[mask] - start_sec

    # Adjust sections
    if track.sections is not None:
        trimmed.sections = []
        for s in track.sections:
            if s.end > start_sec and s.start < end_sec:
                trimmed.sections.append(Section(
                    label=s.label,
                    start=max(0.0, s.start - start_sec),
                    end=min(end_sec - start_sec, s.end - start_sec),
                    energy=s.energy,
                ))

    # Trim stems if present
    if track.stems is not None:
        trimmed.stems = copy.copy(track.stems)
        if track.stems.vocals is not None:
            trimmed.stems.vocals = track.stems.vocals[start_sample:end_sample].copy()
        if track.stems.drums is not None:
            trimmed.stems.drums = track.stems.drums[start_sample:end_sample].copy()
        if track.stems.bass is not None:
            trimmed.stems.bass = track.stems.bass[start_sample:end_sample].copy()
        if track.stems.other is not None:
            trimmed.stems.other = track.stems.other[start_sample:end_sample].copy()

    # Reset loudness so it gets re-measured on the segment
    trimmed.loudness_lufs = None

    return trimmed


def prepare_segments(
    track_a: Track, track_b: Track, is_first: bool = False,
) -> tuple[Track, Track, object]:
    """Prepare trimmed segments for a transition between two tracks.

    If is_first is True, track A starts from the beginning (DJ opener).

    The algorithm determines:
    1. WHERE to exit track A (after its last high-energy section)
    2. WHERE to enter track B (before its first chorus/drop)
    3. If there's a lyric match, adjusts exit/entry to align on the shared word

    Returns (segment_a, segment_b, lyric_cue) — trimmed Track copies ready
    for strategies, plus the lyric cue (or None) so strategies can use it
    directly without re-searching.
    """
    duration_a = track_a.duration_seconds or (len(track_a.audio) / track_a.sample_rate)
    duration_b = track_b.duration_seconds or (len(track_b.audio) / track_b.sample_rate)

    # --- Step 1: Structure-based exit/entry ---
    if is_first:
        # First track in the set should start from the top.
        a_start = 0.0
    else:
        # Later transitions should continue from the region where this track
        # would typically be introduced, reducing repeated content.
        a_start, _ = find_entry_point(track_a)

    a_end = find_exit_point(track_a)
    exit_time = a_end  # Save for layer zone calculation
    b_start, b_target_section = find_entry_point(track_b)
    b_end = duration_b

    # Extend A into the post-exit transition zone.
    # Prefer a bridge if present; otherwise include the first section after
    # the exit point so instrumental-first transitions have room to build.
    if track_a.sections:
        bridge_end = None
        first_after_exit_end = None
        for s in track_a.sections:
            if s.start < exit_time:
                continue
            if first_after_exit_end is None:
                first_after_exit_end = s.end
            if s.label == "bridge":
                bridge_end = s.end
                break

        if bridge_end is not None:
            a_end = max(a_end, bridge_end)
            logger.info("Extended A segment through bridge (to %.1fs)", bridge_end)
        elif first_after_exit_end is not None:
            a_end = max(a_end, first_after_exit_end)
            logger.info("Extended A segment through post-exit section (to %.1fs)",
                        first_after_exit_end)

    # Keep ranges valid and ensure we have enough material.
    a_start = min(max(0.0, a_start), duration_a)
    a_end = min(max(a_start, a_end), duration_a)
    b_start = min(max(0.0, b_start), duration_b)

    if a_end - a_start < 10.0:
        a_end = min(duration_a, a_start + 30.0)
    if a_end - a_start < 1.0:
        a_start = max(0.0, duration_a - 10.0)
        a_end = duration_a

    # --- Step 2: Try lyric-based refinement ---
    lyric_cue = None
    if track_a.lyrics is not None and track_b.lyrics is not None:
        from mixing.lyric_sync import find_lyric_cues

        # Look for shared words near the planned exit/entry zones
        a_window = (max(0.0, a_end - 30.0), min(duration_a, a_end + 15.0))
        b_window = (max(0.0, b_start), min(duration_b, b_start + 30.0))

        cues = find_lyric_cues(
            track_a.lyrics, track_b.lyrics,
            a_exit_window=a_window,
            b_entry_window=b_window,
            title_a=track_a.title,
            title_b=track_b.title,
        )

        if cues:
            import dataclasses as _dc

            lyric_cue = cues[0]

            # B enters at the chorus/drop start found by find_entry_point —
            # this is the first energetic section and is more reliable than
            # trying to find which section contains the cue word.
            if b_target_section is not None:
                lyric_cue = _dc.replace(
                    lyric_cue, b_phrase_start=b_target_section.start,
                )
                logger.info("B entry: start of %s @ %.1fs",
                             b_target_section.label, b_target_section.start)

            # Extend segment through the bridge after the exit point.
            # Bridges are natural transition zones — the stem layer strategy
            # uses the bridge to layer B's instrumental under A.
            cue_end = lyric_cue.a_end + ENTRY_RUNWAY_SEC
            layer_zone_end = exit_time
            if track_a.sections:
                bridge_end = None
                first_after_exit = None
                for s in track_a.sections:
                    if s.start >= exit_time:
                        if first_after_exit is None:
                            first_after_exit = s.end
                        if s.label == "bridge":
                            bridge_end = s.end
                            break
                if bridge_end is not None:
                    layer_zone_end = bridge_end
                elif first_after_exit is not None:
                    layer_zone_end = first_after_exit
            a_end = min(max(cue_end, layer_zone_end), duration_a)
            # Keep B from the entry point (usually 0) so the full intro
            # is available.  Don't let the lyric cue trim B's start —
            # strategies need the intro material for layering.
            b_start = min(b_start, max(0.0, lyric_cue.b_start - ENTRY_RUNWAY_SEC))

            logger.info("Lyric cue: transition on \"%s\" (A@%.1fs -> B@%.1fs, conf=%.2f)",
                         lyric_cue.word, lyric_cue.a_start, lyric_cue.b_start,
                         lyric_cue.confidence)

    logger.info("Track A: %s [%.1f-%.1fs = %.1fs]%s",
                 track_a.title, a_start, a_end, a_end - a_start,
                 f" (lyric: \"{lyric_cue.word}\")" if lyric_cue else "")
    logger.info("Track B: %s [%.1f-%.1fs = %.1fs]%s",
                 track_b.title, b_start, b_end, b_end - b_start,
                 f" (entering on \"{lyric_cue.word}\")" if lyric_cue else "")

    seg_a = extract_segment(track_a, a_start, a_end)
    seg_b = extract_segment(track_b, b_start, b_end)

    # Adjust lyric cue times from original-track coordinates to segment coordinates
    if lyric_cue is not None:
        import dataclasses as _dc
        lyric_cue = _dc.replace(
            lyric_cue,
            a_start=lyric_cue.a_start - a_start,
            a_end=lyric_cue.a_end - a_start,
            b_start=lyric_cue.b_start - b_start,
            b_end=lyric_cue.b_end - b_start,
            b_phrase_start=lyric_cue.b_phrase_start - b_start,
        )

    return seg_a, seg_b, lyric_cue

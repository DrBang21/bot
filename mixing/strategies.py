"""Transition strategies: each strategy defines eligibility, scoring, and execution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from config import config
from models.track import Track
from mixing.bpm_utils import (
    bpm_difference_pct,
    bars_to_samples,
    bars_to_seconds,
)
from mixing.crossfade import apply_crossfade
from mixing.beat_alignment import align_tracks
from processing.effects import (
    echo_out, vinyl_brake, scratch_in,
    hpf_sweep, lpf_sweep,
    split_bands, extract_loop, tile_loop,
)
from processing.loudness import normalize_loudness
from processing.timestretch import time_stretch_audio

logger = logging.getLogger("dj.mixing.strategies")

# Maximum BPM difference (%) for crossfade eligibility (after double/half time)
CROSSFADE_BPM_THRESHOLD = 0.05  # 5%


class TransitionStrategy(ABC):
    """Base class for all transition strategies."""

    name: str = "base"

    @abstractmethod
    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        """Can this transition type work between these two tracks?"""

    @abstractmethod
    def score(self, track_a: Track, track_b: Track) -> float:
        """How well does this transition fit? Higher = better (0.0 to 1.0)."""

    @abstractmethod
    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        """Execute the transition. Returns the combined audio array."""

    def apply_with_cue(self, track_a: Track, track_b: Track, lyric_cue: object) -> np.ndarray:
        """Execute with a pre-found lyric cue. Override in strategies that use it."""
        return self.apply(track_a, track_b)


class EchoOutTransition(TransitionStrategy):
    """Echo/delay tail on track A, then bring in track B.

    Good for energy shifts or when other techniques don't fit.
    """

    name = "echo_out"

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        return True

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.5
        diff = bpm_difference_pct(track_a.bpm, track_b.bpm)
        # Best for moderate-to-large BPM gaps where beat-matched
        # techniques can't work cleanly
        if diff > 0.15:
            return 0.75
        if diff > CROSSFADE_BPM_THRESHOLD:
            return 0.70
        return 0.40  # For close BPMs, stem layer / EQ mix are better

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        # Echo out the tail of track A (configurable bars at its BPM)
        echo_bars = config.echo_bars
        echo_bpm = track_a.bpm or 120.0
        echo_duration = bars_to_seconds(echo_bpm, echo_bars)
        delay_sec = 60.0 / echo_bpm  # Delay = one beat for rhythmic echo

        audio_a_processed = echo_out(
            track_a.audio, config.sample_rate, echo_duration,
            delay_sec=delay_sec, feedback=0.45, wet=0.6,
        )

        # Bring in track B from its first downbeat
        b_beats = track_b.downbeats if track_b.downbeats is not None else track_b.beats
        cut_b = int(b_beats[0] * track_b.sample_rate) if b_beats is not None and len(b_beats) > 0 else 0
        head = track_b.audio[cut_b:].copy()

        # Overlap the echo tail with B's entrance (1 bar crossfade)
        overlap_samples = bars_to_samples(track_b.bpm or 120.0, 1, config.sample_rate)
        overlap_samples = min(overlap_samples, len(head), len(audio_a_processed))
        if overlap_samples <= 0:
            return np.concatenate([audio_a_processed, head])

        # Fade out A's echo tail
        audio_a_processed[-overlap_samples:] *= np.linspace(
            1.0, 0.0, overlap_samples, dtype=np.float32)
        # Fade in B
        head[:overlap_samples] *= np.linspace(
            0.0, 1.0, overlap_samples, dtype=np.float32)

        # Overlap-add the transition zone
        a_before = audio_a_processed[:-overlap_samples]
        blended = audio_a_processed[-overlap_samples:] + head[:overlap_samples]
        b_after = head[overlap_samples:]

        result = np.concatenate([a_before, blended, b_after])
        logger.info("Echo-out transition (%.1fs echo, delay=%.2fs)", echo_duration, delay_sec)
        return result


class VinylBrakeTransition(TransitionStrategy):
    """Vinyl brake (motor-off) on track A, then drop track B.

    Great for dramatic energy shifts — the classic turntable power-down.
    """

    name = "vinyl_brake"

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        return True

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.3
        diff = bpm_difference_pct(track_a.bpm, track_b.bpm)
        # Vinyl brake is dramatic — best for big BPM/energy shifts
        if diff > 0.15:
            return 0.8
        if diff > CROSSFADE_BPM_THRESHOLD:
            return 0.55
        return 0.2  # Feels over-the-top for small BPM gaps

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        audio_a_processed = vinyl_brake(
            track_a.audio,
            config.sample_rate,
            duration_sec=config.spinback_duration,
        )

        # Drop in B from first downbeat with a short fade-in
        b_beats = track_b.downbeats if track_b.downbeats is not None else track_b.beats
        cut_b = int(b_beats[0] * track_b.sample_rate) if b_beats is not None and len(b_beats) > 0 else 0
        head = track_b.audio[cut_b:].copy()

        # Tiny fade-in on B to smooth the drop
        fade_in = min(int(0.05 * config.sample_rate), len(head))
        if fade_in > 0:
            head[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)

        result = np.concatenate([audio_a_processed, head])
        logger.info("Vinyl brake transition (%.1fs)", config.spinback_duration)
        return result


class ScratchInTransition(TransitionStrategy):
    """Vinyl scratch-in: tease the incoming track with rhythmic scratches
    over the outgoing track, then drop into the full track.

    Needs beat data on track B to extract a scratch stab. Always eligible
    but scores highest for dramatic transitions.
    """

    name = "scratch_in"

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        # Need beat data on B to extract a meaningful stab
        if track_b.beats is None or len(track_b.beats) < 2:
            return False
        if track_a.bpm is None:
            return False
        return True

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.25
        diff = bpm_difference_pct(track_a.bpm, track_b.bpm)
        # Scratch-in is a showmanship move — works for any gap but
        # especially fun for moderate-to-big shifts
        if diff > 0.15:
            return 0.7
        if diff > CROSSFADE_BPM_THRESHOLD:
            return 0.6
        return 0.3  # For close BPMs, smoother techniques are better

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        scratch_bpm = track_a.bpm or 120.0

        # Extract a short sample from track B's first beat for scratching
        stab = extract_loop(track_b.audio, track_b.beats, config.sample_rate,
                            beat_index=0, n_beats=1)

        # Generate 4-scratch pattern timed to track A's BPM
        scratch_audio = scratch_in(stab, config.sample_rate, scratch_bpm,
                                   n_scratches=4)

        # Layer scratches over the tail of track A
        scratch_len = len(scratch_audio)
        a_end = len(track_a.audio)
        layer_start = max(0, a_end - scratch_len)

        audio_a = track_a.audio.copy()
        mix_len = min(scratch_len, a_end - layer_start)
        audio_a[layer_start:layer_start + mix_len] += scratch_audio[:mix_len] * 0.6

        # Fade out A over the last bar
        fade_out_samples = bars_to_samples(scratch_bpm, 1, config.sample_rate)
        fade_out_samples = min(fade_out_samples, len(audio_a))
        if fade_out_samples > 0:
            audio_a[-fade_out_samples:] *= np.linspace(
                1.0, 0.0, fade_out_samples, dtype=np.float32
            )

        # Drop in B from first downbeat with a short crossfade overlap
        b_beats = track_b.downbeats if track_b.downbeats is not None else track_b.beats
        cut_b = int(b_beats[0] * track_b.sample_rate) if b_beats is not None and len(b_beats) > 0 else 0
        head = track_b.audio[cut_b:].copy()

        # Overlap last bit of A (with scratches fading) and B entrance
        overlap = min(int(0.1 * config.sample_rate), len(head), len(audio_a))
        if overlap <= 0:
            return np.concatenate([audio_a, head])

        audio_a[-overlap:] *= np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        head[:overlap] *= np.linspace(0.0, 1.0, overlap, dtype=np.float32)

        a_before = audio_a[:-overlap]
        blended = audio_a[-overlap:] + head[:overlap]
        b_after = head[overlap:]

        result = np.concatenate([a_before, blended, b_after])
        logger.info("Scratch-in transition: 4 scratches at %.1f BPM", scratch_bpm)
        return result


class FilterMixTransition(TransitionStrategy):
    """Filter mix: HPF sweep out on track A, LPF filtered intro on track B.

    Requires beat-matched BPMs (same threshold as crossfade) so the
    filtered sections overlap cleanly.
    """

    name = "filter_mix"

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        if track_a.bpm is None or track_b.bpm is None:
            return False
        if track_a.beats is None or track_b.beats is None:
            return False
        return bpm_difference_pct(track_a.bpm, track_b.bpm) <= CROSSFADE_BPM_THRESHOLD

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.0
        diff = bpm_difference_pct(track_a.bpm, track_b.bpm)
        # Slightly below crossfade — it's more interesting but also riskier
        return max(0.0, 0.9 - diff * 8.0)

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        offset_a, offset_b = align_tracks(track_a, track_b)

        matched_bpm = track_a.bpm or 120.0
        fade_samples = bars_to_samples(matched_bpm, config.transition_bars, config.sample_rate)

        # HPF sweep out on A (remove lows progressively)
        audio_a = track_a.audio.copy()
        audio_a = hpf_sweep(audio_a, config.sample_rate, fade_samples,
                            start_hz=20.0, end_hz=5000.0)
        audio_a = audio_a[:offset_a + fade_samples]

        # LPF filtered intro on B (start muffled, open up)
        audio_b = track_b.audio[offset_b:].copy()
        audio_b = lpf_sweep(audio_b, config.sample_rate, fade_samples,
                            start_hz=20000.0, end_hz=200.0)

        # Blend with equal-power crossfade over the filter region
        result = apply_crossfade(audio_a, audio_b, fade_samples)
        logger.info("Filter mix: %d bars HPF out + LPF in at %.1f BPM",
                     config.transition_bars, matched_bpm)
        return result


class EQMixTransition(TransitionStrategy):
    """EQ/frequency mix: swap low/mid/high bands progressively.

    Strip the bass from the outgoing track while bringing in the bass of
    the incoming track, let the hi-hats blend in the middle. Requires
    beat-matched BPMs.
    """

    name = "eq_mix"

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        if track_a.bpm is None or track_b.bpm is None:
            return False
        if track_a.beats is None or track_b.beats is None:
            return False
        return bpm_difference_pct(track_a.bpm, track_b.bpm) <= CROSSFADE_BPM_THRESHOLD

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.0
        diff = bpm_difference_pct(track_a.bpm, track_b.bpm)
        # EQ mix is the pro technique — scores high for close BPMs
        return max(0.0, 0.95 - diff * 8.0)

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        offset_a, offset_b = align_tracks(track_a, track_b)

        matched_bpm = track_a.bpm or 120.0
        fade_samples = bars_to_samples(matched_bpm, config.transition_bars, config.sample_rate)

        # Extract transition regions
        region_a = track_a.audio[offset_a:offset_a + fade_samples].copy()
        region_b = track_b.audio[offset_b:offset_b + fade_samples].copy()

        # Pad if needed
        if len(region_a) < fade_samples:
            region_a = np.pad(region_a, (0, fade_samples - len(region_a)))
        if len(region_b) < fade_samples:
            region_b = np.pad(region_b, (0, fade_samples - len(region_b)))

        # Split into frequency bands
        a_low, a_mid, a_high = split_bands(region_a, config.sample_rate)
        b_low, b_mid, b_high = split_bands(region_b, config.sample_rate)

        # Create crossfade envelopes
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = 1.0 - fade_in

        # Swap bands progressively:
        # - Lows swap first (bass swap is the foundation)
        # - Mids swap in the middle
        # - Highs swap last (hi-hats blend longest)
        third = fade_samples // 3

        # Lows: swap in first third
        low_fade_in = np.zeros(fade_samples, dtype=np.float32)
        low_fade_in[third:] = 1.0
        low_fade_in[:third] = np.linspace(0.0, 1.0, third, dtype=np.float32)
        lows = a_low * (1.0 - low_fade_in) + b_low * low_fade_in

        # Mids: swap in middle third
        mids = a_mid * fade_out + b_mid * fade_in

        # Highs: swap in last third
        high_fade_in = np.zeros(fade_samples, dtype=np.float32)
        high_fade_in[:2 * third] = np.linspace(0.0, 0.5, 2 * third, dtype=np.float32)
        high_fade_in[2 * third:] = np.linspace(0.5, 1.0, fade_samples - 2 * third, dtype=np.float32)
        highs = a_high * (1.0 - high_fade_in) + b_high * high_fade_in

        blended = lows + mids + highs

        # Assemble: [A before] + [blended] + [B after]
        a_before = track_a.audio[:offset_a]
        b_after = track_b.audio[offset_b + fade_samples:]

        result = np.concatenate([a_before, blended, b_after])
        logger.info("EQ mix: band-swapped over %d bars at %.1f BPM",
                     config.transition_bars, matched_bpm)
        return result


class LoopInTransition(TransitionStrategy):
    """Loop in: create a 4-beat loop from the incoming track, blend over it,
    then release into the full track.

    Requires beat-matched BPMs and beat data on track B.
    """

    name = "loop_in"

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        if track_a.bpm is None or track_b.bpm is None:
            return False
        if track_b.beats is None or len(track_b.beats) < 5:
            return False
        return bpm_difference_pct(track_a.bpm, track_b.bpm) <= CROSSFADE_BPM_THRESHOLD

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.0
        diff = bpm_difference_pct(track_a.bpm, track_b.bpm)
        # Creative technique — slightly lower than EQ mix
        return max(0.0, 0.85 - diff * 8.0)

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        # Time-stretch B to match A if needed
        offset_a, offset_b = align_tracks(track_a, track_b)

        matched_bpm = track_a.bpm or 120.0

        # Extract a 4-beat loop from the start of track B
        loop = extract_loop(track_b.audio, track_b.beats, config.sample_rate,
                            beat_index=0, n_beats=4)

        # Repeat the loop for 8 bars (= 32 beats = 8 repetitions of 4-beat loop)
        n_repeats = config.transition_bars * 4 // 4  # bars * beats_per_bar / loop_beats
        looped = tile_loop(loop, n_repeats)

        # Fade in the loop over first 2 bars
        fade_in_len = bars_to_samples(matched_bpm, 2, config.sample_rate)
        fade_in_len = min(fade_in_len, len(looped))
        looped[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)

        # Mix the loop under track A's tail
        loop_samples = len(looped)
        a_tail_start = max(0, len(track_a.audio) - loop_samples)
        a_tail = track_a.audio[a_tail_start:].copy()

        # Fade out A over the loop region
        a_fade_out = np.linspace(1.0, 0.0, len(a_tail), dtype=np.float32)
        a_tail *= a_fade_out

        # Trim loop to match A tail length
        mix_len = min(len(a_tail), len(looped))
        blended = a_tail[:mix_len] + looped[:mix_len]

        # After the loop, release into full track B from the offset
        b_release = track_b.audio[offset_b:]

        a_before = track_a.audio[:a_tail_start]
        result = np.concatenate([a_before, blended, b_release])
        logger.info("Loop-in transition: 4-beat loop x%d, release into %s",
                     n_repeats, track_b.title)
        return result


class StemLayerTransition(TransitionStrategy):
    """Stem-layered transition with optional lyric cue.

    1. Track A plays at its BPM (full mix).
    2. Track B's instrumental is time-stretched to A's BPM and layered in
       (faded or instant) under A's vocals.
    3. At the switch point (lyric cue or end of layer zone), cut to Track B's
       full mix at B's original BPM.

    The BPM shift is hidden because only the instrumental is stretched — no
    vocals are distorted. At the switch, B snaps to its natural BPM.

    Requires stems on both tracks.
    """

    name = "stem_layer"

    def __init__(self, fade_in_instrumental: bool = True):
        self.fade_in_instrumental = fade_in_instrumental

    def is_eligible(self, track_a: Track, track_b: Track) -> bool:
        if track_a.bpm is None or track_b.bpm is None:
            return False
        if track_a.stems is None or track_b.stems is None:
            return False
        # Need vocals + instrumental on A and instrumental on B.
        if track_a.stems.vocals is None:
            return False
        if track_a.stems.instrumental is None:
            return False
        if track_b.stems.instrumental is None:
            return False
        return True

    def score(self, track_a: Track, track_b: Track) -> float:
        if not track_a.bpm or not track_b.bpm:
            return 0.0
        # This is the premium transition — always scores high when available.
        # Even higher if there's a lyric cue (the decider doesn't know about
        # the cue yet, but the segment selector will have found one if it exists).
        base = 0.85
        if track_a.lyrics and track_b.lyrics:
            # Quick check — do they share any words at all?
            common = track_a.lyrics.find_common_words(track_b.lyrics)
            if common:
                base = 0.95
        return base

    def apply(self, track_a: Track, track_b: Track) -> np.ndarray:
        """Apply without a lyric cue — switch at end of layer zone."""
        return self._apply_internal(track_a, track_b, lyric_cue=None)

    def apply_with_cue(self, track_a: Track, track_b: Track, lyric_cue: object) -> np.ndarray:
        """Apply with the pre-found lyric cue from the segment selector."""
        return self._apply_internal(track_a, track_b, lyric_cue=lyric_cue)

    def _apply_internal(self, track_a: Track, track_b: Track, lyric_cue) -> np.ndarray:
        if len(track_a.audio) == 0:
            return track_b.audio.copy()
        if len(track_b.audio) == 0:
            return track_a.audio.copy()

        normalize_loudness(track_a)
        normalize_loudness(track_b)

        sr = config.sample_rate
        bpm_a = track_a.bpm or 120.0
        bpm_b = track_b.bpm or 120.0

        if lyric_cue is not None:
            logger.info("Stem layer using lyric cue: \"%s\" (A@%.1fs, B@%.1fs)",
                         lyric_cue.word, lyric_cue.a_start, lyric_cue.b_start)

        # --- Section-driven timing ---
        duration_a = track_a.duration_seconds or len(track_a.audio) / sr

        # Find the best section for the layer zone.
        # Rule: for same-BPM tracks, transition during the SECOND chorus.
        # This lets the listener hear through the first chorus and verse
        # before the blend starts.  When structure detection mislabels a
        # chorus as "verse", we fall back to section position + energy.
        exit_section = None
        if track_a.sections:
            energies = [s.energy for s in track_a.sections]
            e_threshold = float(np.percentile(energies, 40)) if energies else 0

            # Collect all chorus/drop sections before the outro tail
            choruses = [s for s in track_a.sections
                        if s.label in ("chorus", "drop")
                        and s.end <= duration_a * 0.90]

            if len(choruses) >= 2:
                # Pick the chorus closest to the 45-55% sweet spot.
                # This consistently targets the second song chorus even
                # when structure detection mislabels a bridge as chorus.
                target_pos = duration_a * 0.50
                exit_section = min(choruses,
                                   key=lambda s: abs(s.start - target_pos))
                logger.info("Exit: chorus nearest 50%% @ %.1fs-%.1fs",
                            exit_section.start, exit_section.end)
            elif len(choruses) == 1:
                # Only one chorus detected — structure detection may have
                # missed the second one.  Look for a section in the 40-70%
                # range with similar energy (likely a mislabeled chorus).
                first_chorus = choruses[0]
                for s in track_a.sections:
                    if s is first_chorus or s.label in ("intro", "outro"):
                        continue
                    rel_start = s.start / duration_a
                    if (0.35 <= rel_start <= 0.70
                            and s.energy >= first_chorus.energy * 0.80):
                        exit_section = s
                        logger.info("Exit: likely second chorus (labeled %s) "
                                     "%.1fs-%.1fs (E=%.2f vs chorus E=%.2f)",
                                     s.label, s.start, s.end,
                                     s.energy, first_chorus.energy)
                        break
                # If no candidate found, use the first chorus
                if exit_section is None:
                    exit_section = first_chorus
            else:
                # No choruses at all — pick a high-energy section after 40%
                for s in reversed(track_a.sections):
                    if (s.end <= duration_a * 0.85
                            and s.end >= duration_a * 0.40
                            and s.label not in ("intro", "outro")
                            and s.energy >= e_threshold):
                        exit_section = s
                        break

        # --- Layer zone selection ---
        # For same-BPM tracks: layer B during the exit section (chorus)
        # itself — this is the core same-BPM transition: A vocals over B
        # instrumental during the chorus, then hard switch.
        # Bridge-style transitions are only for different-BPM tracks where
        # the energy valley helps mask the tempo change.
        max_layer_dur = bars_to_seconds(bpm_a, config.transition_bars)
        use_bridge_style = False

        if exit_section is not None:
            # No bridge — layer B during the exit section itself.
            # Use the last max_layer_dur of the exit section so the
            # switch lands right at the section boundary.
            switch_a_sec = exit_section.end
            section_dur = exit_section.end - exit_section.start
            if section_dur > max_layer_dur:
                layer_start_sec = exit_section.end - max_layer_dur
            else:
                layer_start_sec = exit_section.start
            logger.info("Layer zone: %s %.1fs-%.1fs (layer during exit section)",
                        exit_section.label, layer_start_sec, switch_a_sec)
        else:
            default_layer_samples = bars_to_samples(bpm_a, config.transition_bars, sr)
            layer_start_sec = max(0.0, duration_a - (default_layer_samples / sr) - 1.0)
            switch_a_sec = layer_start_sec + max_layer_dur

        # Lyric cue switch — only for non-bridge transitions.
        # Bridge transitions always switch at bridge end for a clean handoff.
        if not use_bridge_style:
            min_layer_for_cue = max_layer_dur * 0.5
            if (lyric_cue is not None
                    and lyric_cue.a_start >= layer_start_sec
                    and lyric_cue.a_start <= switch_a_sec
                    and (lyric_cue.a_start - layer_start_sec) >= min_layer_for_cue):
                switch_a_sec = lyric_cue.a_start
                logger.info("Lyric cue \"%s\" at %.1fs in layer zone — switching on cue",
                             lyric_cue.word, lyric_cue.a_start)
            elif (lyric_cue is not None
                  and lyric_cue.a_start >= layer_start_sec
                  and lyric_cue.a_start <= switch_a_sec):
                logger.info("Lyric cue \"%s\" at %.1fs too early in layer (need %.1fs) — ignoring",
                             lyric_cue.word, lyric_cue.a_start, min_layer_for_cue)

        switch_a_sec = min(switch_a_sec, duration_a)

        # --- Beat-align layer_start so B's first beat lands on A's grid ---
        # B's audio starts at layer_start in the mix.  B's first beat is
        # at B_phase seconds into B's audio.  For the beats to align,
        # layer_start + B_phase must coincide with an A beat.
        # So: layer_start = A_beat - B_phase
        if track_a.beats is not None and len(track_a.beats) > 0:
            beats = track_a.beats

            # B's first beat: where the music actually starts in B's audio.
            # layer_start + b_first_beat must land on an A beat so the
            # two tracks are grid-aligned.
            b_first_beat = getattr(track_b, '_first_beat', None) or 0.0

            # Find the first A beat AT OR AFTER (layer_start + b_first_beat).
            target = layer_start_sec + b_first_beat
            future_beats = beats[beats >= target]
            if len(future_beats) > 0:
                aligned_start = float(future_beats[0]) - b_first_beat
                logger.info("Layer start beat-aligned: %.2fs -> %.2fs "
                             "(A beat %.3fs - B first_beat %.3fs)",
                             layer_start_sec, aligned_start,
                             float(future_beats[0]), b_first_beat)
                layer_start_sec = aligned_start

            # Snap switch to the nearest beat at or before
            past_switch = beats[beats <= switch_a_sec + 0.05]
            if len(past_switch) > 0:
                snapped = float(past_switch[-1])
                if snapped >= layer_start_sec + 2.0:
                    logger.info("Switch point snapped to A beat: %.2fs -> %.2fs",
                                 switch_a_sec, snapped)
                    switch_a_sec = snapped

        layer_start = max(0, int(layer_start_sec * sr))
        switch_a_sample = min(int(switch_a_sec * sr), len(track_a.audio))
        phase2_len = switch_a_sample - layer_start

        # --- Phase 1: Track A plays normally up to layer zone ---
        phase1 = track_a.audio[:layer_start].copy()

        # --- Phase 2: A vocals + B full audio ---
        # At the chorus start, A's instrumental cuts completely.
        # B starts playing from the beginning — its intro is ~14s of
        # pure instrumental (no vocals), so no clash.
        # A's vocal stem plays on top at full volume.
        # Only the very last word of A fades out.
        b_zone = track_b.audio[:phase2_len].copy()
        if len(b_zone) < phase2_len:
            b_zone = np.pad(b_zone, (0, phase2_len - len(b_zone)))

        mix_len = min(phase2_len, len(b_zone))

        if (track_a.stems is not None
                and track_a.stems.vocals is not None):
            a_vocals_zone = track_a.stems.vocals[layer_start:switch_a_sample].copy()
            a_vocals_zone = a_vocals_zone[:mix_len]
            # A vocals play completely untouched — no fade at all.
            phase2 = a_vocals_zone + b_zone[:mix_len]
            logger.info("Phase 2: A vocals + B full audio (%.1fs)", mix_len / sr)
        else:
            phase2 = track_a.audio[layer_start:switch_a_sample].copy() + b_zone[:mix_len]

        # --- Phase 3: B continues from where Phase 2 left off ---
        # No switch logic needed — B just keeps playing.
        phase3 = track_b.audio[phase2_len:].copy()

        # Micro-fade at the seam for click prevention
        micro = min(int(0.005 * sr), len(phase2), max(1, len(phase3)))
        if micro > 0 and len(phase2) >= micro:
            phase2[-micro:] *= np.linspace(1.0, 0.0, micro, dtype=np.float32)
        if micro > 0 and len(phase3) >= micro:
            phase3[:micro] *= np.linspace(0.0, 1.0, micro, dtype=np.float32)

        result = np.concatenate([phase1, phase2, phase3])

        if lyric_cue:
            logger.info("Stem layer transition: %d bars instrumental layer, "
                         "switch on \"%s\" — B enters from phrase @ %.1fs "
                         "(A@%.1fs -> B@%.1fs, B reverts to %.1f BPM)",
                         config.transition_bars, lyric_cue.word,
                         lyric_cue.b_phrase_start,
                         lyric_cue.a_start, lyric_cue.b_start, bpm_b)
        else:
            logger.info("Stem layer transition: %d bars instrumental layer, "
                         "switch at %.1fs (B at %.1f BPM)",
                         config.transition_bars, switch_a_sec, bpm_b)
        return result


# Registry of all available strategies
ALL_STRATEGIES: list[TransitionStrategy] = [
    EchoOutTransition(),
    VinylBrakeTransition(),
    ScratchInTransition(),
    FilterMixTransition(),
    EQMixTransition(),
    LoopInTransition(),
    StemLayerTransition(fade_in_instrumental=True),
]

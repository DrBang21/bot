import logging

import librosa
import numpy as np

from models.track import Track

logger = logging.getLogger("dj.analysis.beats")


def _normalize(x: np.ndarray) -> np.ndarray:
    """Return x scaled to 0..1 (safe for all-zero arrays)."""
    x = np.asarray(x, dtype=np.float32)
    max_val = float(np.max(np.abs(x))) if x.size else 0.0
    if max_val <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return x / max_val


def _choose_downbeat_phase(
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    low_band_env: np.ndarray,
) -> int:
    """Choose the best 4-beat phase to treat as downbeats.

    Beat tracking gives quarter-note pulses, but not bar starts. We score
    each phase (0, 1, 2, 3) by onset and low-frequency strength and keep
    the strongest one as the likely downbeat phase.
    """
    if len(beat_frames) < 8:
        return 0

    n = min(len(onset_env), len(low_band_env))
    if n == 0:
        return 0

    onset = onset_env[:n]
    bass = low_band_env[:n]

    best_phase = 0
    best_score = -np.inf
    for phase in range(4):
        phase_frames = beat_frames[phase::4]
        if len(phase_frames) == 0:
            continue
        idx = np.clip(phase_frames.astype(int), 0, n - 1)
        onset_score = float(np.mean(onset[idx]))
        bass_score = float(np.mean(bass[idx]))
        # Slightly favor onset transients but still bias to bass impact.
        score = 0.65 * onset_score + 0.35 * bass_score
        if score > best_score:
            best_score = score
            best_phase = phase

    return best_phase


def detect_beats(track: Track) -> Track:
    """Detect beat positions and set track.beats (times in seconds)."""
    if track.audio is None or len(track.audio) == 0:
        raise ValueError("Track audio must be loaded before beat detection")

    hop_length = 512
    onset_env = librosa.onset.onset_strength(
        y=track.audio,
        sr=track.sample_rate,
        hop_length=hop_length,
    )
    onset_env = _normalize(onset_env)

    # Low-band envelope helps anchor bars on kick-heavy beats.
    stft = np.abs(librosa.stft(track.audio, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=track.sample_rate, n_fft=2048)
    low_mask = freqs <= 180.0
    if np.any(low_mask):
        low_band_env = np.mean(stft[low_mask], axis=0)
    else:
        low_band_env = np.mean(stft, axis=0)
    low_band_env = _normalize(low_band_env)

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=track.sample_rate,
        hop_length=hop_length,
        trim=False,
    )
    beat_frames = np.asarray(beat_frames, dtype=np.int64)

    if beat_frames.size == 0:
        track.beats = np.array([], dtype=np.float32)
        track.downbeats = np.array([], dtype=np.float32)
        logger.warning("No beats detected for %s", track.title)
        return track

    beat_times = librosa.frames_to_time(
        beat_frames, sr=track.sample_rate, hop_length=hop_length
    )
    track.beats = beat_times

    # Estimate downbeats (every 4 beats, assuming 4/4 time) with phase alignment.
    if len(beat_times) >= 4:
        phase = _choose_downbeat_phase(beat_frames, onset_env, low_band_env)
        track.downbeats = beat_times[phase::4]
    else:
        phase = 0
        track.downbeats = beat_times

    logger.info(
        "Detected %d beats (%d downbeats, phase=%d) for %s",
        len(beat_times),
        len(track.downbeats),
        phase,
        track.title,
    )
    return track


def refine_beats(track: Track) -> Track:
    """Refine beat/downbeat positions using the onset-optimal BPM grid.

    After detect_bpm runs, track._bpm_phase and track.bpm give us a
    mathematically perfect grid that aligns with actual onset transients.
    This grid is more precise than librosa's detected beat positions,
    which can drift by ~100ms due to spectral estimation bias.

    For each existing downbeat, snap it to the nearest grid point.  This
    keeps the structural alignment (which beats are "downbeats") from
    librosa's phase detection while fixing the timing precision.
    """
    if (track._bpm_phase is None or track.bpm is None
            or track.downbeats is None or len(track.downbeats) == 0
            or track.audio is None):
        return track

    duration = track.duration_seconds or (len(track.audio) / track.sample_rate)
    interval = 60.0 / track.bpm
    phase = track._bpm_phase

    # Build the precise beat grid from onset-correlation
    precise_beats = np.arange(phase, duration, interval)
    # Build precise downbeat grid (every 4 beats)
    precise_downbeats = np.arange(phase, duration, interval * 4)

    if len(precise_beats) < 4 or len(precise_downbeats) < 2:
        return track

    # Snap each detected downbeat to the nearest precise downbeat
    refined_downbeats = []
    for db in track.downbeats:
        diffs = np.abs(precise_downbeats - db)
        nearest_idx = int(np.argmin(diffs))
        snapped = float(precise_downbeats[nearest_idx])
        # Only snap if reasonably close (within half a beat interval)
        if diffs[nearest_idx] < interval * 0.5:
            refined_downbeats.append(snapped)
        else:
            refined_downbeats.append(float(db))

    old_db = track.downbeats
    track.downbeats = np.array(refined_downbeats, dtype=np.float64)
    track.beats = precise_beats

    # Log the max correction applied
    corrections = np.abs(track.downbeats[:len(old_db)] - old_db[:len(track.downbeats)])
    if len(corrections) > 0:
        max_corr = float(np.max(corrections))
        mean_corr = float(np.mean(corrections))
        logger.info(
            "Refined %d downbeats for %s (max correction: %.0fms, mean: %.0fms)",
            len(track.downbeats), track.title,
            max_corr * 1000, mean_corr * 1000,
        )

    return track

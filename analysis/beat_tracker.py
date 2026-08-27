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
) -> tuple[int, float]:
    """Choose the best 4-beat phase to treat as downbeats.

    Beat tracking gives quarter-note pulses, but not bar starts. We score
    each phase (0, 1, 2, 3) by onset and low-frequency strength and keep
    the strongest one as the likely downbeat phase.
    """
    if len(beat_frames) < 8:
        return 0, 0.0

    n = min(len(onset_env), len(low_band_env))
    if n == 0:
        return 0, 0.0

    onset = onset_env[:n]
    bass = low_band_env[:n]

    phase_scores: list[float] = []
    for phase in range(4):
        phase_frames = beat_frames[phase::4]
        if len(phase_frames) == 0:
            phase_scores.append(float("-inf"))
            continue
        idx = np.clip(phase_frames.astype(int), 0, n - 1)
        onset_score = float(np.mean(onset[idx]))
        bass_score = float(np.mean(bass[idx]))
        # Slightly favor onset transients but still bias to bass impact.
        score = 0.65 * onset_score + 0.35 * bass_score
        phase_scores.append(score)

    best_phase = int(np.argmax(phase_scores))
    finite = sorted((s for s in phase_scores if np.isfinite(s)), reverse=True)
    if len(finite) < 2 or finite[0] <= 0:
        confidence = 0.0
    else:
        # Relative winning margin: 0 means ambiguous, 1 means decisive.
        confidence = max(0.0, min(1.0, (finite[0] - finite[1]) / finite[0]))
    return best_phase, confidence


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
        track.downbeat_phase = None
        track.downbeat_confidence = 0.0
        logger.warning("No beats detected for %s", track.title)
        return track

    beat_times = librosa.frames_to_time(
        beat_frames, sr=track.sample_rate, hop_length=hop_length
    )
    track.beats = beat_times

    # Estimate downbeats (every 4 beats, assuming 4/4 time) with phase alignment.
    if len(beat_times) >= 4:
        phase, confidence = _choose_downbeat_phase(beat_frames, onset_env, low_band_env)
        track.downbeats = beat_times[phase::track.meter]
    else:
        phase = 0
        confidence = 0.0
        track.downbeats = beat_times
    track.downbeat_phase = phase
    track.downbeat_confidence = confidence

    logger.info(
        "Detected %d beats (%d downbeats, phase=%d, confidence=%.2f) for %s",
        len(beat_times),
        len(track.downbeats),
        phase,
        confidence,
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
    if len(precise_beats) < track.meter:
        return track

    # Anchor the refined bar grid to the originally detected musical
    # downbeat.  The BPM phase is only a quarter-note phase; treating it as
    # beat one is what caused correctly synced beats but shifted bars.
    old_db = np.asarray(track.downbeats, dtype=np.float64)
    anchor = float(old_db[0])
    anchor_idx = int(np.argmin(np.abs(precise_beats - anchor)))
    while anchor_idx - track.meter >= 0:
        anchor_idx -= track.meter
    track.downbeats = precise_beats[anchor_idx::track.meter].astype(np.float64)
    track.beats = precise_beats

    # Report fixed-grid timing drift independently from bar phase. Comparing
    # old downbeats only with new downbeats can show nearly one bar when the
    # two estimators accumulated tempo drift, which looked like a phase bug.
    nearest_beats = np.array([
        precise_beats[np.argmin(np.abs(precise_beats - db))] for db in old_db
    ])
    corrections = np.abs(nearest_beats - old_db)
    if len(corrections) > 0:
        max_corr = float(np.max(corrections))
        mean_corr = float(np.mean(corrections))
        logger.info(
            "Refined bar grid for %s (%d downbeats; fixed-grid beat drift: "
            "max %.0fms, mean %.0fms)",
            track.title, len(track.downbeats),
            max_corr * 1000, mean_corr * 1000,
        )

    return track

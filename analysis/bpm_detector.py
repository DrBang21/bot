import logging

import librosa
import numpy as np

from config import config
from models.track import Track

logger = logging.getLogger("dj.analysis.bpm")


def _onset_correlation_bpm(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    candidates: list[float],
) -> tuple[float, float, dict[float, float]]:
    """Score candidate BPMs by how well a perfect grid aligns with onsets.

    For each candidate BPM, tests multiple phase offsets and keeps the
    best-scoring phase.  Returns (best_bpm, best_phase_sec, {bpm: score}).
    """
    n_frames = len(onset_env)
    duration = librosa.frames_to_time(n_frames, sr=sr, hop_length=hop_length)

    scores: dict[float, float] = {}
    best_global_bpm = candidates[0]
    best_global_score = 0.0
    best_global_phase = 0.0

    for bpm in candidates:
        interval_sec = 60.0 / bpm
        n_phases = max(1, int(interval_sec * 1000 / 5))
        best_score = 0.0
        best_phase = 0.0

        for phase_idx in range(n_phases):
            phase_sec = phase_idx * 0.005
            beat_times = np.arange(phase_sec, duration, interval_sec)
            beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
            beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < n_frames)]

            if len(beat_frames) < 4:
                continue

            score = float(np.mean(onset_env[beat_frames]))
            if score > best_score:
                best_score = score
                best_phase = phase_sec

        scores[bpm] = best_score
        if best_score > best_global_score:
            best_global_score = best_score
            best_global_bpm = bpm
            best_global_phase = best_phase

    return best_global_bpm, best_global_phase, scores


def _find_first_beat(audio: np.ndarray, sr: int, phase: float) -> float:
    """Find the time of the first rhythmic transient in the audio.

    Scans onset strength to find the first significant transient, then
    snaps it to the nearest beat on the BPM grid.  This gives the actual
    position where the music starts rhythmically, aligned to the grid.

    This is what a DJ does visually in a DAW: find where the waveform
    first shows a beat hit, and place that on the grid.
    """
    hop = 128  # high resolution
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    if len(onset_env) == 0:
        return phase

    # Threshold: 30% of the max onset in the first 3 seconds.
    limit = min(len(onset_env), int(3.0 * sr / hop))
    max_onset = float(np.max(onset_env[:limit]))
    threshold = max_onset * 0.30

    # Find the first onset peak above threshold
    first_transient = None
    for i in range(1, limit - 1):
        if (onset_env[i] > onset_env[i - 1]
                and onset_env[i] > onset_env[i + 1]
                and onset_env[i] >= threshold):
            first_transient = float(onset_times[i])
            break

    if first_transient is None:
        return phase

    return first_transient


def detect_bpm(track: Track) -> Track:
    """Estimate BPM using onset-correlation scoring and set track.bpm.

    Also detects the time of the first beat in the audio (_bpm_first_beat)
    so that downstream code can align tracks precisely by matching their
    first beats to the beat grid.
    """
    if track.audio is None or len(track.audio) == 0:
        raise ValueError("Track audio must be loaded before BPM detection")

    hop_length = 512

    # Get rough spectral estimate to narrow the search range
    tempo_bt, _ = librosa.beat.beat_track(
        y=track.audio,
        sr=track.sample_rate,
        start_bpm=120.0,
    )
    bpm_bt = float(np.atleast_1d(tempo_bt)[0])

    tempo_ac = librosa.feature.tempo(
        y=track.audio,
        sr=track.sample_rate,
        start_bpm=120.0,
    )
    bpm_ac = float(np.atleast_1d(tempo_ac)[0])

    spectral_avg = (bpm_bt + bpm_ac) / 2.0

    # Compute onset envelope for correlation scoring
    onset_env = librosa.onset.onset_strength(
        y=track.audio,
        sr=track.sample_rate,
        hop_length=hop_length,
    )

    # Generate integer candidates
    center = round(spectral_avg)
    candidates = [float(c) for c in range(center - 1, center + 3)]

    best_bpm, best_phase, scores = _onset_correlation_bpm(
        onset_env, track.sample_rate, hop_length, candidates,
    )

    # Clamp to configured range
    if best_bpm < config.min_bpm:
        best_bpm *= 2
    elif best_bpm > config.max_bpm:
        best_bpm /= 2

    track.bpm = round(best_bpm)

    # Store the grid phase (for building precise beat arrays)
    track._bpm_phase = best_phase

    # Find the first actual beat in the audio — this is where the music
    # starts rhythmically.  Used for aligning B's audio in transitions.
    first_beat = _find_first_beat(track.audio, track.sample_rate, best_phase)
    track._first_beat = first_beat

    scores_str = ", ".join(f"{int(b)}={s:.4f}" for b, s in sorted(scores.items()))
    logger.info("Detected BPM: %d for %s (first_beat=%.3fs, phase=%.3fs, "
                "spectral: %.2f/%.2f, scores: %s)",
                track.bpm, track.title, first_beat, best_phase,
                bpm_bt, bpm_ac, scores_str)
    return track

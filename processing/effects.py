"""Audio effects for DJ transitions: echo out, spinback, filters, EQ, loops."""

import logging

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger("dj.processing.effects")


def echo_out(audio: np.ndarray, sr: int, duration_sec: float,
             delay_sec: float = 0.25, feedback: float = 0.5,
             wet: float = 0.6) -> np.ndarray:
    """Apply an echo/delay tail that fades out over duration_sec.

    The echo is applied to the last `duration_sec` of the audio and the
    signal is faded to silence.
    """
    if len(audio) == 0:
        return audio.copy()

    n_samples = int(duration_sec * sr)
    n_samples = min(n_samples, len(audio))
    if n_samples <= 0:
        return audio.copy()

    delay_samples = max(1, int(delay_sec * sr))

    tail = audio[-n_samples:].copy()

    # Build echo via simple comb filter
    echoed = np.zeros(len(tail) + delay_samples * 6, dtype=np.float32)
    echoed[:len(tail)] = tail
    for tap in range(1, 7):
        offset = delay_samples * tap
        gain = feedback ** tap
        if offset < len(echoed):
            end = min(offset + len(tail), len(echoed))
            echoed[offset:end] += tail[:end - offset] * gain

    # Trim to original tail length
    echoed = echoed[:n_samples]

    # Mix wet/dry
    tail_mixed = tail * (1.0 - wet) + echoed * wet

    # Apply fade-out envelope
    fade = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
    tail_mixed *= fade

    result = audio.copy()
    result[-n_samples:] = tail_mixed
    logger.info("Applied echo-out effect (%.1fs, delay=%.2fs)", duration_sec, delay_sec)
    return result


def vinyl_brake(audio: np.ndarray, sr: int, duration_sec: float = 1.5) -> np.ndarray:
    """Apply a vinyl brake (motor-off decelerate-to-stop) to the last duration_sec.

    Simulates cutting the turntable motor — playback decelerates and pitch drops.
    """
    if len(audio) == 0:
        return audio.copy()

    n_samples = int(duration_sec * sr)
    n_samples = min(n_samples, len(audio))
    if n_samples <= 1:
        return audio.copy()

    tail = audio[-n_samples:]

    # Create a time-warping curve: starts at normal speed, decelerates to zero
    # Using quadratic deceleration for realistic feel
    t_norm = np.linspace(0.0, 1.0, n_samples)
    # Phase accumulation: integral of (1 - t^2) gives t - t^3/3
    phase = t_norm - (t_norm ** 3) / 3.0
    phase /= phase[-1]  # Normalize to [0, 1]
    # Map phase to sample indices in the tail
    indices = phase * (n_samples - 1)
    indices = np.clip(indices.astype(int), 0, n_samples - 1)

    warped = tail[indices]

    # Apply a fade-out to silence
    fade = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
    warped = warped * fade

    result = audio.copy()
    result[-n_samples:] = warped
    logger.info("Applied vinyl brake effect (%.1fs)", duration_sec)
    return result


# ---------------------------------------------------------------------------
# Filter utilities
# ---------------------------------------------------------------------------

def _butter_filter(audio: np.ndarray, sr: int, cutoff: float,
                   btype: str, order: int = 4) -> np.ndarray:
    """Apply a Butterworth filter (lowpass or highpass)."""
    if len(audio) == 0:
        return audio.copy()

    nyq = sr / 2.0
    cutoff = max(20.0, min(cutoff, nyq - 1.0))
    sos = butter(order, cutoff / nyq, btype=btype, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def hpf_sweep(audio: np.ndarray, sr: int, n_samples: int,
              start_hz: float = 20.0, end_hz: float = 5000.0) -> np.ndarray:
    """Progressively apply a high-pass filter sweep over the last n_samples.

    Sweeps cutoff from start_hz to end_hz, removing lows over time.
    Processed in chunks for a smooth sweep.
    """
    n_samples = min(n_samples, len(audio))
    if n_samples <= 0:
        return audio.copy()

    n_chunks = 32
    chunk_size = max(1, n_samples // n_chunks)

    result = audio.copy()
    freqs = np.geomspace(start_hz, end_hz, n_chunks)

    for i, freq in enumerate(freqs):
        start = len(audio) - n_samples + i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(audio)
        result[start:end] = _butter_filter(result[start:end], sr, freq, "highpass")

    logger.info("Applied HPF sweep (%.0f -> %.0f Hz over %d samples)",
                start_hz, end_hz, n_samples)
    return result


def lpf_sweep(audio: np.ndarray, sr: int, n_samples: int,
              start_hz: float = 20000.0, end_hz: float = 200.0) -> np.ndarray:
    """Progressively apply a low-pass filter sweep over the first n_samples.

    Sweeps cutoff from start_hz down to end_hz, then back up — creating
    a filtered intro that opens up.
    """
    n_samples = min(n_samples, len(audio))
    if n_samples <= 0:
        return audio.copy()

    n_chunks = 32
    chunk_size = max(1, n_samples // n_chunks)

    result = audio.copy()
    # Sweep down then back up (V-shape)
    freqs_down = np.geomspace(start_hz, end_hz, n_chunks // 2)
    freqs_up = np.geomspace(end_hz, start_hz, n_chunks - n_chunks // 2)
    freqs = np.concatenate([freqs_down, freqs_up])

    for i, freq in enumerate(freqs):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n_samples
        result[start:end] = _butter_filter(result[start:end], sr, freq, "lowpass")

    logger.info("Applied LPF sweep (%.0f -> %.0f -> %.0f Hz over %d samples)",
                start_hz, end_hz, start_hz, n_samples)
    return result


# ---------------------------------------------------------------------------
# EQ band splitting
# ---------------------------------------------------------------------------

def split_bands(audio: np.ndarray, sr: int,
                low_cut: float = 300.0,
                high_cut: float = 3000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split audio into low, mid, and high frequency bands.

    Returns (lows, mids, highs) — three arrays that sum back to ~original.
    """
    lows = _butter_filter(audio, sr, low_cut, "lowpass")
    highs = _butter_filter(audio, sr, high_cut, "highpass")
    mids = audio - lows - highs  # Residual = mids
    return lows, mids, highs


# ---------------------------------------------------------------------------
# Loop utilities
# ---------------------------------------------------------------------------

def extract_loop(audio: np.ndarray, beats: np.ndarray, sr: int,
                 beat_index: int = 0, n_beats: int = 4) -> np.ndarray:
    """Extract an n_beats loop starting at beat_index.

    Returns a seamlessly loopable segment.
    """
    if beats is None or len(beats) < beat_index + n_beats + 1:
        # Fallback: estimate from beat spacing
        if beats is not None and len(beats) >= 2:
            beat_dur = np.median(np.diff(beats))
        else:
            beat_dur = 0.5  # Assume 120 BPM
        start_sample = int(beats[beat_index] * sr) if beats is not None and len(beats) > beat_index else 0
        length = int(n_beats * beat_dur * sr)
        return audio[start_sample:start_sample + length].copy()

    start = int(beats[beat_index] * sr)
    end = int(beats[beat_index + n_beats] * sr)
    return audio[start:end].copy()


def tile_loop(loop: np.ndarray, n_repeats: int) -> np.ndarray:
    """Repeat a loop segment n times."""
    return np.tile(loop, n_repeats)


# ---------------------------------------------------------------------------
# Scratch utilities
# ---------------------------------------------------------------------------

def scratch_in(sample: np.ndarray, sr: int, bpm: float,
               n_scratches: int = 4) -> np.ndarray:
    """Generate a vinyl scratch-in pattern from a short audio sample.

    Creates a rhythmic forward-reverse-forward scratch pattern timed to
    the beat, simulating a DJ teasing the incoming track.

    Each scratch consists of:
      - Forward stab (quarter-beat, pitched up slightly from speed)
      - Reverse pull-back (eighth-beat, quieter)
      - Silence gap to the next beat

    Returns the full scratch sequence ready to layer over the outgoing track.
    """
    if bpm <= 0:
        raise ValueError("bpm must be > 0 for scratch generation")
    if n_scratches <= 0:
        return np.array([], dtype=np.float32)
    if len(sample) == 0:
        return np.zeros(int((60.0 / bpm) * sr) * n_scratches, dtype=np.float32)

    beat_sec = 60.0 / bpm
    beat_samples = int(beat_sec * sr)
    if beat_samples <= 0:
        return np.array([], dtype=np.float32)

    # Use just the first ~quarter-beat of the sample as the scratch stab
    stab_len = min(int(beat_samples * 0.25), len(sample))
    stab = sample[:stab_len].copy()

    # Apply a short attack/release envelope to avoid clicks
    env_len = min(64, stab_len // 4)
    if env_len > 0:
        stab[:env_len] *= np.linspace(0.0, 1.0, env_len, dtype=np.float32)
        stab[-env_len:] *= np.linspace(1.0, 0.0, env_len, dtype=np.float32)

    # Reverse version (pull-back) — shorter and quieter
    reverse = stab[::-1].copy() * 0.5
    reverse_len = len(reverse) // 2
    reverse = reverse[:reverse_len]

    # Build each scratch: [stab] + [reverse] + [silence to next beat]
    scratches = []
    for i in range(n_scratches):
        # Progressively increase volume (teasing → full reveal)
        gain = 0.3 + 0.7 * (i / max(n_scratches - 1, 1))

        scratch_block = np.zeros(beat_samples, dtype=np.float32)
        # Place stab at start of beat
        scratch_block[:stab_len] = stab * gain
        # Place reverse right after
        rev_start = stab_len
        rev_end = min(rev_start + reverse_len, beat_samples)
        scratch_block[rev_start:rev_end] = reverse[:rev_end - rev_start] * gain

        scratches.append(scratch_block)

    result = np.concatenate(scratches) if scratches else np.array([], dtype=np.float32)
    logger.info("Generated scratch-in: %d scratches at %.1f BPM (%.1fs)",
                n_scratches, bpm, len(result) / sr)
    return result

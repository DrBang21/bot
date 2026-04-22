import logging

import numpy as np

logger = logging.getLogger("dj.mixing.crossfade")


def linear_curve(length: int) -> np.ndarray:
    """Generate a linear fade curve from 0 to 1."""
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def equal_power_curve(length: int) -> np.ndarray:
    """Generate an equal-power (sinusoidal) fade curve from 0 to 1."""
    return np.sin(np.linspace(0.0, np.pi / 2, length)).astype(np.float32)


def apply_crossfade(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    fade_samples: int,
    curve: str = "equal_power",
) -> np.ndarray:
    """Crossfade between the tail of audio_a and the head of audio_b.

    Returns a single combined audio array:
      [audio_a before fade] + [crossfaded region] + [audio_b after fade]
    """
    if len(audio_a) == 0:
        return audio_b.copy()
    if len(audio_b) == 0:
        return audio_a.copy()

    fade_samples = min(fade_samples, len(audio_a), len(audio_b))
    if fade_samples <= 0:
        result = np.concatenate([audio_a, audio_b])
        logger.info("Applied %s crossfade over 0 samples (hard join fallback)", curve)
        return result

    if curve == "linear":
        fade_in = linear_curve(fade_samples)
    else:
        fade_in = equal_power_curve(fade_samples)
    fade_out = fade_in[::-1]

    # Regions
    a_before = audio_a[:-fade_samples]
    a_fade = audio_a[-fade_samples:]
    b_fade = audio_b[:fade_samples]
    b_after = audio_b[fade_samples:]

    # Blend the crossfade region
    blended = a_fade * fade_out + b_fade * fade_in

    result = np.concatenate([a_before, blended, b_after])
    logger.info("Applied %s crossfade over %d samples", curve, fade_samples)
    return result

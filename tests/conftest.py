import numpy as np
import pytest


@pytest.fixture
def sine_wave():
    """Generate a 5-second 440Hz sine wave at 44100Hz for testing."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr

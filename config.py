from dataclasses import dataclass


@dataclass
class DJConfig:
    sample_rate: int = 44100
    target_lufs: float = -14.0
    min_bpm: float = 60.0
    max_bpm: float = 200.0
    output_format: str = "wav"
    audio_dir: str = "audio"

    # Transition settings
    transition_bars: int = 8       # Crossfade length in bars (at matched BPM)
    echo_bars: int = 4             # Echo-out tail length in bars
    spinback_duration: float = 1.5 # Spinback duration in seconds


config = DJConfig()

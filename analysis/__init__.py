import logging

from analysis.beat_tracker import detect_beats
from analysis.bpm_detector import detect_bpm
from analysis.key_detector import detect_key
from analysis.structure_analyzer import detect_structure
from models.track import Track

logger = logging.getLogger("dj.analysis")


def analyze(track: Track, separate: bool = True, lyrics: bool = True) -> Track:
    """Run full analysis pipeline: BPM, beats, key, structure, stems, and lyrics."""
    detect_beats(track)
    detect_bpm(track)
    detect_key(track)
    detect_structure(track)

    if separate:
        try:
            from analysis.stem_separator import separate_stems

            separate_stems(track)
        except ModuleNotFoundError as exc:
            logger.warning(
                "Stem separation skipped (%s). Install torch/demucs to enable stems.",
                exc,
            )
        except Exception:
            logger.exception("Stem separation failed; continuing without stems")

    if lyrics:
        try:
            from analysis.lyric_detector import detect_lyrics

            track.lyrics = detect_lyrics(track)
        except ModuleNotFoundError as exc:
            logger.warning(
                "Lyric detection skipped (%s). Install faster-whisper to enable lyrics.",
                exc,
            )
        except Exception:
            logger.exception("Lyric detection failed; continuing without lyrics")

    if track.lyrics is not None and track.sections:
        from analysis.structure_analyzer import refine_sections_with_lyrics
        refine_sections_with_lyrics(track)

    return track

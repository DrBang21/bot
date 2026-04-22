"""Lyric detection using faster-whisper on vocal stems.

Runs speech-to-text on the separated vocal track to get word-level
timestamps. This enables lyric-matching transitions (e.g. transition
on a shared word between two tracks).
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from models.track import Track

logger = logging.getLogger("dj.analysis.lyrics")

# Lazy-loaded model singleton
_model = None


@dataclass
class Word:
    """A single detected word with its timestamp."""
    text: str
    start: float   # seconds
    end: float      # seconds
    probability: float = 0.0


@dataclass
class Lyrics:
    """All detected lyrics for a track."""
    words: list[Word] = field(default_factory=list)
    full_text: str = ""

    def find_word(self, target: str) -> list[Word]:
        """Find all occurrences of a word (case-insensitive)."""
        target_lower = target.lower().strip()
        return [w for w in self.words if w.text.lower().strip() == target_lower]

    def find_phrase(self, phrase: str) -> list[tuple[float, float]]:
        """Find approximate occurrences of a multi-word phrase.

        Returns list of (start_time, end_time) for each match.
        """
        phrase_words = phrase.lower().strip().split()
        if not phrase_words:
            return []

        matches = []
        word_texts = [w.text.lower().strip() for w in self.words]

        for i in range(len(word_texts) - len(phrase_words) + 1):
            if word_texts[i:i + len(phrase_words)] == phrase_words:
                matches.append((
                    self.words[i].start,
                    self.words[i + len(phrase_words) - 1].end,
                ))

        return matches

    def find_common_words(self, other: "Lyrics", min_length: int = 3) -> list[str]:
        """Find words that appear in both lyrics (ignoring short/common words).

        Returns unique matching words sorted by length (longest first —
        longer shared words make better transition cues).
        """
        # Common filler words to ignore
        stopwords = {
            "the", "and", "but", "for", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "are", "has", "his", "how",
            "its", "let", "may", "new", "now", "old", "see", "way", "who",
            "did", "get", "got", "him", "just", "like", "that", "this",
            "with", "will", "from", "have", "been", "they", "them", "then",
            "than", "what", "when", "your", "yeah", "yah", "ooh", "uhh",
        }

        words_a = {
            w.text.lower().strip()
            for w in self.words
            if len(w.text.strip()) >= min_length
        }
        words_b = {
            w.text.lower().strip()
            for w in other.words
            if len(w.text.strip()) >= min_length
        }

        common = (words_a & words_b) - stopwords
        return sorted(common, key=len, reverse=True)


def _get_model():
    """Lazy-load the Whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model (small)...")
        # "small" gives much better accuracy on music vocals than "base".
        # Prefer CUDA when available, otherwise fall back to CPU.
        device = "cpu"
        compute_type = "int8"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception:
            # torch may be unavailable when stem separation isn't installed.
            pass

        try:
            _model = WhisperModel("small", device=device, compute_type=compute_type)
        except ValueError:
            _model = WhisperModel("small", device=device, compute_type="float32")

        logger.info("Whisper model loaded (%s)", device.upper())
    return _model


def detect_lyrics(track: Track) -> Lyrics:
    """Run speech-to-text on the vocal stem to detect lyrics with timestamps.

    Requires track.stems.vocals to be populated (run stem separation first).
    Falls back to full audio if vocals aren't available.
    """
    if track.stems is not None and track.stems.vocals is not None:
        audio = track.stems.vocals
        logger.info("Running Whisper on vocal stem for %s", track.title)
    else:
        audio = track.audio
        logger.info("Running Whisper on full audio for %s (no vocal stem)", track.title)

    if audio is None:
        logger.warning("No audio available for lyric detection on %s", track.title)
        return Lyrics()

    # faster-whisper needs a file path — write vocals to a temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, track.sample_rate)
        tmp_path = tmp.name

    try:
        model = _get_model()

        segments, _ = model.transcribe(
            tmp_path,
            word_timestamps=True,
            language="en",
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,  # Prevents hallucination loops
            repetition_penalty=1.1,            # Penalise repeated tokens slightly
            no_speech_threshold=0.5,           # More aggressive silence detection
            vad_filter=True,  # Silero VAD filters non-speech, reduces hallucination
            vad_parameters={"min_silence_duration_ms": 300},
        )

        words = []
        full_parts = []

        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append(Word(
                        text=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=float(w.probability or 0.0),
                    ))
                    full_parts.append(w.word)

        full_text = " ".join(full_parts).strip()

        lyrics = Lyrics(words=words, full_text=full_text)
        logger.info("Detected %d words for %s: \"%s\"",
                     len(words), track.title,
                     full_text[:80] + "..." if len(full_text) > 80 else full_text)
        return lyrics

    finally:
        Path(tmp_path).unlink(missing_ok=True)

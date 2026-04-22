"""Lyric sync: finds shared words between two tracks and returns transition
cue points that create wordplay transitions.

Example: Track A says "right back", Track B says "we back" — transition on
the word "back" so it sounds like a play on words.
"""

import logging
from dataclasses import dataclass

from analysis.lyric_detector import Lyrics, Word

logger = logging.getLogger("dj.mixing.lyric_sync")

# Phrase detection: a gap longer than this between words = new phrase
_PHRASE_GAP_SEC = 0.4
# Don't look back more than this from the cue word for phrase start
_PHRASE_MAX_LOOKBACK_SEC = 4.0


@dataclass
class LyricCue:
    """A matched word/phrase that could serve as a transition point."""
    word: str
    # Where the word appears in track A (outgoing)
    a_start: float  # seconds
    a_end: float
    # Where the word appears in track B (incoming)
    b_start: float
    b_end: float
    # Start of the phrase containing the cue word in B (e.g. "we back..." starts at "we")
    b_phrase_start: float = 0.0
    # Confidence (higher = better match)
    confidence: float = 0.0


def _find_phrase_start(lyrics: Lyrics, word_time: float) -> float:
    """Find the start of the lyric phrase containing the word at word_time.

    Walks backwards through B's words looking for a gap > _PHRASE_GAP_SEC,
    which indicates a phrase/line boundary. Caps lookback at _PHRASE_MAX_LOOKBACK_SEC
    so we don't start too far before the cue word.
    """
    # Find the word index closest to the given time
    target_idx = None
    for i, w in enumerate(lyrics.words):
        if abs(w.start - word_time) < 0.15:
            target_idx = i
            break

    if target_idx is None:
        return word_time

    # Walk backwards to find phrase boundary
    phrase_start_idx = target_idx
    for i in range(target_idx - 1, -1, -1):
        gap = lyrics.words[i + 1].start - lyrics.words[i].end
        if gap > _PHRASE_GAP_SEC:
            break
        if word_time - lyrics.words[i].start > _PHRASE_MAX_LOOKBACK_SEC:
            break
        phrase_start_idx = i

    return lyrics.words[phrase_start_idx].start


def find_lyric_cues(
    lyrics_a: Lyrics, lyrics_b: Lyrics,
    a_exit_window: tuple[float, float] | None = None,
    b_entry_window: tuple[float, float] | None = None,
    title_a: str = "",
    title_b: str = "",
) -> list[LyricCue]:
    """Find shared words between two tracks that could serve as transition cues.

    Optionally constrain to time windows (e.g., only look for matches near
    the planned exit/entry points).

    If title_a / title_b are provided, words that appear in either track's
    title get a significant confidence boost — these are the most meaningful
    wordplay candidates (e.g. "back" in "Right Back" + "We Back").

    Returns cues sorted by confidence (best first).
    """
    if not lyrics_a.words or not lyrics_b.words:
        return []

    # Find common words between the two tracks
    common = lyrics_a.find_common_words(lyrics_b)
    if not common:
        logger.info("No common words found between tracks")
        return []

    logger.info("Common words: %s", common[:10])

    # Build set of title words for boosting
    title_words = set()
    for title in (title_a, title_b):
        for w in title.lower().split():
            # Strip common filename suffixes
            cleaned = w.strip("()-_.,!?")
            if len(cleaned) >= 3:
                title_words.add(cleaned)

    if title_words:
        logger.info("Title words for boosting: %s", title_words)

    cues = []

    for word in common:
        a_hits = lyrics_a.find_word(word)
        b_hits = lyrics_b.find_word(word)

        if not a_hits or not b_hits:
            continue

        # Filter to time windows if provided
        if a_exit_window is not None:
            a_hits = [
                w for w in a_hits
                if a_exit_window[0] <= w.start <= a_exit_window[1]
            ]
        if b_entry_window is not None:
            b_hits = [
                w for w in b_hits
                if b_entry_window[0] <= w.start <= b_entry_window[1]
            ]

        if not a_hits or not b_hits:
            continue

        for a_word in a_hits:
            for b_word in b_hits:
                # Base confidence: word length + whisper probability
                word_len_score = min(len(word) / 8.0, 1.0)
                prob_score = (a_word.probability + b_word.probability) / 2.0
                confidence = word_len_score * 0.4 + prob_score * 0.3

                # Big boost if the word appears in either track's title
                if word.lower() in title_words:
                    confidence += 0.3

                # Find the start of the phrase in B ("we back..." → "we")
                phrase_start = _find_phrase_start(lyrics_b, b_word.start)

                cues.append(LyricCue(
                    word=word,
                    a_start=a_word.start,
                    a_end=a_word.end,
                    b_start=b_word.start,
                    b_end=b_word.end,
                    b_phrase_start=phrase_start,
                    confidence=confidence,
                ))

    # Sort by confidence (best first)
    cues.sort(key=lambda c: c.confidence, reverse=True)

    if cues:
        best = cues[0]
        logger.info("Best lyric cue: \"%s\" (A@%.1fs, B@%.1fs, conf=%.2f)",
                     best.word, best.a_start, best.b_start, best.confidence)

    return cues

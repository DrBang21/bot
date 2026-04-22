"""RAG-enhanced transition decider.

Builds a natural-language description of the track pair, retrieves the most
relevant knowledge from the vector store, and uses it to adjust the base
strategy scores from the rule-based decider.

Score adjustments
-----------------
- Matching *rules* give a relevance-weighted boost to the named strategy.
- Matching *feedback* shifts scores up (high rating) or down (low rating).
- The base score from the rule-based system is always the starting point;
  RAG only nudges — it cannot invent new strategies.
"""

from __future__ import annotations

import logging

from knowledge.vector_store import TransitionKnowledgeBase
from mixing.transition_decider import get_eligible_transitions, TransitionOption
from models.track import Track

logger = logging.getLogger("dj.mixing.rag")


# ------------------------------------------------------------------
# Query builder — turns a track pair into a retrieval query
# ------------------------------------------------------------------

def _build_query(track_a: Track, track_b: Track) -> str:
    """Create a natural-language description of the track pair for retrieval."""
    parts: list[str] = []

    # -- Track A structure --
    if track_a.sections:
        labels = [s.label for s in track_a.sections]
        parts.append(f"Track A structure: {', '.join(labels)}")
        if "bridge" in labels:
            parts.append("Track A has a bridge section")
        if "drop" in labels:
            parts.append("Track A has a drop")
        parts.append(f"Track A ends with {labels[-1]}")

    # -- Track B structure --
    if track_b.sections:
        labels = [s.label for s in track_b.sections]
        parts.append(f"Track B structure: {', '.join(labels)}")
        parts.append(f"Track B starts with {labels[0]}")

    # -- BPM relationship --
    if track_a.bpm and track_b.bpm:
        diff = abs(track_a.bpm - track_b.bpm) / max(track_a.bpm, track_b.bpm)
        if diff < 0.03:
            parts.append("BPM is very similar between both tracks")
        elif diff < 0.08:
            parts.append("BPM is somewhat different between tracks")
        else:
            parts.append("BPM is significantly different between tracks")

    # -- Energy comparison --
    if track_a.sections and track_b.sections:
        a_avg = sum(s.energy for s in track_a.sections) / len(track_a.sections)
        b_avg = sum(s.energy for s in track_b.sections) / len(track_b.sections)
        if b_avg > a_avg * 1.2:
            parts.append("Track B has higher energy than track A")
        elif a_avg > b_avg * 1.2:
            parts.append("Track A has higher energy than track B")
        else:
            parts.append("Both tracks have similar energy levels")

    # -- Key --
    if track_a.key and track_b.key:
        if track_a.key == track_b.key:
            parts.append(f"Both tracks are in {track_a.key}")
        else:
            parts.append(f"Track A is in {track_a.key}, track B is in {track_b.key}")

    # -- Stems availability --
    if track_a.stems is not None and track_b.stems is not None:
        parts.append("Both tracks have separated stems available")

    return ". ".join(parts)


# ------------------------------------------------------------------
# RAG-adjusted selection
# ------------------------------------------------------------------

def rag_select(
    track_a: Track,
    track_b: Track,
    kb: TransitionKnowledgeBase,
) -> TransitionOption | None:
    """Pick the best transition using base scores + RAG adjustments.

    Returns *None* only when no strategy is eligible at all.
    """
    options = get_eligible_transitions(track_a, track_b)
    if not options:
        return None

    # Shortcut when the KB is empty (first run before any feedback)
    if kb.count() == 0:
        best = options[0]
        logger.info("RAG: no knowledge yet — using base score: %s (%.2f)",
                     best.strategy.name, best.score)
        return best

    # Retrieve relevant knowledge
    query = _build_query(track_a, track_b)
    results = kb.query(query, n_results=5)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        return options[0]

    # Log what we retrieved
    for doc, meta, dist in zip(docs, metas, dists):
        logger.debug("  Retrieved (d=%.3f, %s): %s",
                      dist, meta.get("type", "?"), doc[:90])

    # Build per-strategy score adjustments
    adjustments: dict[str, float] = {}
    for doc, meta, dist in zip(docs, metas, dists):
        relevance = max(0.0, 1.0 - dist)  # cosine distance → similarity
        strategy_name = meta.get("strategy", "")
        entry_type = meta.get("type", "rule")

        if not strategy_name or strategy_name == "general":
            continue

        if strategy_name not in adjustments:
            adjustments[strategy_name] = 0.0

        if entry_type == "feedback":
            # Rating 1-5 → meaningful adjustment scaled by relevance.
            # A single 5/5 on a similar pair should noticeably boost
            # the strategy; a 1/5 should noticeably penalize it.
            # Scale: 1→-0.3, 2→-0.15, 3→0, 4→+0.15, 5→+0.3
            rating = meta.get("rating", 3)
            adj = (rating - 3) * 0.15 * relevance
        else:
            # Rules give a positive boost scaled by relevance
            adj = 0.15 * relevance

        adjustments[strategy_name] += adj

    # Apply adjustments to base scores
    for opt in options:
        adj = adjustments.get(opt.strategy.name, 0.0)
        if abs(adj) > 0.001:
            old = opt.score
            opt.score = max(0.0, min(1.0, opt.score + adj))
            logger.info("RAG adjusted %s: %.2f -> %.2f (delta %+.3f)",
                         opt.strategy.name, old, opt.score, adj)

    # Re-rank after adjustments
    options.sort(key=lambda o: o.score, reverse=True)

    best = options[0]
    logger.info("RAG selected: %s (score: %.2f)", best.strategy.name, best.score)
    return best


# ------------------------------------------------------------------
# Feedback recording
# ------------------------------------------------------------------

def record_feedback(
    kb: TransitionKnowledgeBase,
    track_a: Track,
    track_b: Track,
    strategy_name: str,
    rating: int,
    notes: str = "",
    b_entry_offset_beats: int | None = None,
) -> None:
    """Record user feedback about a completed transition.

    Captures structural features (BPM relationship, section layout, key,
    energy) so that future track pairs with similar characteristics can
    retrieve this feedback even if the track names are completely different.
    TF-IDF retrieval matches on these feature keywords, not track names.
    """
    # --- Build feature-rich text for TF-IDF retrieval ---
    parts: list[str] = []
    parts.append(f"Strategy: {strategy_name}. Rating: {rating}/5.")

    # BPM relationship (most important for retrieval)
    if track_a.bpm and track_b.bpm:
        diff_pct = abs(track_a.bpm - track_b.bpm) / max(track_a.bpm, track_b.bpm)
        if diff_pct < 0.01:
            parts.append(f"Same BPM ({int(track_a.bpm)}).")
        elif diff_pct < 0.03:
            parts.append(f"Similar BPM ({int(track_a.bpm)} vs {int(track_b.bpm)}).")
        else:
            parts.append(f"Different BPM ({int(track_a.bpm)} vs {int(track_b.bpm)}).")

    # Structure summary
    if track_a.sections:
        labels_a = [s.label for s in track_a.sections]
        parts.append(f"Track A sections: {' '.join(labels_a)}.")
        n_choruses = labels_a.count("chorus")
        if n_choruses >= 2:
            parts.append("Track A has multiple choruses.")
        if "bridge" in labels_a:
            parts.append("Track A has bridge.")

    if track_b.sections:
        labels_b = [s.label for s in track_b.sections]
        parts.append(f"Track B sections: {' '.join(labels_b)}.")
        if labels_b[0] == "intro":
            parts.append("Track B has intro before vocals.")

    # Key relationship
    if track_a.key and track_b.key:
        if track_a.key == track_b.key:
            parts.append(f"Same key ({track_a.key}).")
        else:
            parts.append(f"Keys: {track_a.key} to {track_b.key}.")

    # Stems availability
    if track_a.stems and track_b.stems:
        parts.append("Both tracks have stems.")

    # Strategy-specific parameters
    if strategy_name == "stem_layer":
        parts.append("Stem layer transition. A vocals over B instrumental.")
        if track_a.bpm and track_b.bpm and abs(track_a.bpm - track_b.bpm) < 2:
            parts.append("Same BPM stem layer: beat-align B first beat to A grid.")
            parts.append("No time-stretching needed. No vocal fading.")
            parts.append("Transition during second chorus.")

    if b_entry_offset_beats is not None:
        parts.append(f"B entry offset: {b_entry_offset_beats} beats.")

    if notes:
        parts.append(f"User notes: {notes}.")

    # Include track names last (least important for retrieval)
    parts.append(f"Tracks: '{track_a.title}' to '{track_b.title}'.")

    text = " ".join(parts)

    # Metadata for programmatic access
    extra: dict = {
        "track_a": track_a.title,
        "track_b": track_b.title,
    }
    if track_a.bpm and track_b.bpm:
        extra["bpm_a"] = track_a.bpm
        extra["bpm_b"] = track_b.bpm
        extra["same_bpm"] = abs(track_a.bpm - track_b.bpm) < 2
    if b_entry_offset_beats is not None:
        extra["b_entry_offset_beats"] = b_entry_offset_beats
    if track_a.sections:
        labels = [s.label for s in track_a.sections]
        extra["a_has_bridge"] = "bridge" in labels
        extra["a_chorus_count"] = labels.count("chorus")
    if track_b.sections:
        labels = [s.label for s in track_b.sections]
        extra["b_starts_with"] = labels[0]

    kb.add_feedback(text, strategy=strategy_name, rating=rating, metadata=extra)

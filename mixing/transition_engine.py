"""Transition engine: orchestrates transition creation using the strategy system.

When a TransitionKnowledgeBase is provided, the engine uses RAG-enhanced
scoring.  Otherwise it falls back to the pure rule-based auto_select.
"""

from __future__ import annotations

import logging

import numpy as np

from models.track import Track
from mixing.transition_decider import get_eligible_transitions, auto_select

logger = logging.getLogger("dj.mixing.transition")

# Optional — set at startup by main.py so every create_transition call
# benefits from RAG without threading the KB through every caller.
_knowledge_base = None


def set_knowledge_base(kb) -> None:
    """Register a TransitionKnowledgeBase for RAG-enhanced decisions."""
    global _knowledge_base
    _knowledge_base = kb


def create_transition(
    track_a: Track,
    track_b: Track,
    strategy_name: str | None = None,
    lyric_cue: object = None,
) -> np.ndarray:
    """Create a transition between two analyzed tracks.

    If strategy_name is given, uses that specific strategy.
    Otherwise, uses RAG-enhanced selection when a knowledge base is
    registered, falling back to pure score-based auto_select.

    Returns the combined audio array covering the transition.
    """
    if strategy_name is not None:
        options = get_eligible_transitions(track_a, track_b)
        match = [o for o in options if o.strategy.name == strategy_name]
        if not match:
            raise ValueError(
                f"Strategy '{strategy_name}' is not eligible for "
                f"{track_a.title} -> {track_b.title}"
            )
        option = match[0]
    else:
        # Prefer RAG when a knowledge base is available
        if _knowledge_base is not None:
            from mixing.rag_decider import rag_select
            option = rag_select(track_a, track_b, _knowledge_base)
        else:
            option = auto_select(track_a, track_b)
        if option is None:
            raise RuntimeError(
                f"No eligible transition for {track_a.title} -> {track_b.title}"
            )

    logger.info(
        "Mixing: %s -> %s [%s] (score: %.2f)",
        track_a.title, track_b.title, option.strategy.name, option.score,
    )

    # Pass lyric_cue to strategies that accept it
    strategy = option.strategy
    if lyric_cue is not None and hasattr(strategy, 'apply_with_cue'):
        return strategy.apply_with_cue(track_a, track_b, lyric_cue)
    return strategy.apply(track_a, track_b)

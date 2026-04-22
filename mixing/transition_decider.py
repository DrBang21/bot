"""Transition decider: evaluates all strategies and returns ranked eligible options."""

import logging
from dataclasses import dataclass

from models.track import Track
from mixing.strategies import TransitionStrategy, ALL_STRATEGIES

logger = logging.getLogger("dj.mixing.decider")


@dataclass
class TransitionOption:
    """A ranked transition option with its strategy and score."""
    strategy: TransitionStrategy
    score: float


def get_eligible_transitions(
    track_a: Track,
    track_b: Track,
    strategies: list[TransitionStrategy] | None = None,
) -> list[TransitionOption]:
    """Return all eligible transitions ranked by score (best first).

    Each strategy self-reports eligibility and its fit score for this
    specific track pair.
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    options = []
    for strategy in strategies:
        if strategy.is_eligible(track_a, track_b):
            s = strategy.score(track_a, track_b)
            options.append(TransitionOption(strategy=strategy, score=s))

    options.sort(key=lambda o: o.score, reverse=True)

    if options:
        names = ", ".join(f"{o.strategy.name}({o.score:.2f})" for o in options)
        logger.info("Eligible transitions for %s -> %s: %s",
                     track_a.title, track_b.title, names)
    else:
        logger.warning("No eligible transitions for %s -> %s",
                        track_a.title, track_b.title)

    return options


def auto_select(track_a: Track, track_b: Track) -> TransitionOption | None:
    """Automatically pick the best-scoring eligible transition."""
    options = get_eligible_transitions(track_a, track_b)
    if not options:
        return None
    best = options[0]
    logger.info("Auto-selected: %s (score: %.2f)", best.strategy.name, best.score)
    return best

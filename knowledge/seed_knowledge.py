"""Seed rules for the transition knowledge base.

These encode the transition preferences established through iterative feedback.
They are upserted on startup so they're always present but never duplicated.
"""

from __future__ import annotations

import logging

from knowledge.vector_store import TransitionKnowledgeBase

logger = logging.getLogger("dj.knowledge")

SEED_RULES: list[dict] = [
    # --- Bridge-based stem layer (primary preference) ---
    {
        "text": (
            "When track A has a bridge section, use stem_layer transition. "
            "Layer B's instrumental during the bridge — it builds up gradually "
            "and reaches full volume 2 bars before the bridge ends. Switch to "
            "track B vocals at the bridge end. Bridges are natural transition "
            "zones because energy dips, making them ideal for blending."
        ),
        "metadata": {"strategy": "stem_layer", "context": "bridge"},
    },
    {
        "text": (
            "For stem_layer transitions the instrumental should always come in "
            "before the vocals. Layer the instrumental first to build "
            "anticipation, then bring in the full track with vocals at the "
            "switch point."
        ),
        "metadata": {"strategy": "stem_layer", "context": "instrumental_first"},
    },
    {
        "text": (
            "For stem_layer transitions without a bridge, start the "
            "instrumental layer at the end of the last chorus and run through "
            "the following section. Cap at the configured transition bars."
        ),
        "metadata": {"strategy": "stem_layer", "context": "no_bridge"},
    },

    {
        "text": (
            "For stem_layer transitions, track B should enter offset into its "
            "chorus by 2 beats rather than right at the chorus start. This "
            "lands B on a stronger hit and naturally creates vocal overlap "
            "with track A's ending. The offset is adjustable via feedback."
        ),
        "metadata": {"strategy": "stem_layer", "context": "b_entry_offset"},
    },

    # --- General mixing rules ---
    {
        "text": (
            "Never use hard cuts between tracks. Always blend smoothly using "
            "crossfades, stem layering, or effects. Abrupt stops break the "
            "flow and energy of the set."
        ),
        "metadata": {"strategy": "general", "context": "no_cuts"},
    },
    {
        "text": (
            "Track B should enter at a musically meaningful point — the start "
            "of a chorus or verse, snapped to the nearest downbeat. Don't "
            "start B in the middle of a phrase."
        ),
        "metadata": {"strategy": "general", "context": "b_entry"},
    },
    {
        "text": (
            "The first track in a set should play from the beginning to "
            "establish the vibe. Don't cut into the middle of the opening "
            "track."
        ),
        "metadata": {"strategy": "general", "context": "first_track"},
    },
    {
        "text": (
            "Use lyric cues to synchronize transitions when both tracks share "
            "a common word. The shared word creates a natural connection "
            "between the two tracks."
        ),
        "metadata": {"strategy": "stem_layer", "context": "lyric_sync"},
    },

    # --- Strategy-specific guidance ---
    {
        "text": (
            "Never use a raw crossfade between two full tracks as it clashes "
            "vocals and instruments together. Instead, crossfade should only "
            "be used within other transitions — for example, crossfading B's "
            "instrumental under A in a stem layer, then crossfading from the "
            "layer into B's full mix over one bar to bring in vocals smoothly."
        ),
        "metadata": {"strategy": "stem_layer", "context": "no_raw_crossfade"},
    },
    {
        "text": (
            "When there is a significant energy drop between track A and "
            "track B, use echo_out to smooth the gap. The echo creates a "
            "natural decay that bridges the energy difference."
        ),
        "metadata": {"strategy": "echo_out", "context": "energy_drop"},
    },
    {
        "text": (
            "When track A has a low-energy outro and track B has a high-energy "
            "intro or chorus, vinyl_brake on A followed by a drop into B "
            "creates an exciting moment."
        ),
        "metadata": {"strategy": "vinyl_brake", "context": "energy_contrast"},
    },
    {
        "text": (
            "When both tracks have similar BPM (within 3%), use stem_layer "
            "and transition during track A's second chorus. Cut track A's "
            "instrumental and layer B's full instrumental under A's vocals. "
            "Only fade the very last word of A's vocals, then hard-switch "
            "to B. The second chorus is preferred because the listener has "
            "heard the first chorus and verse — the transition feels natural "
            "and B's instrumental fills the gap left by removing A's beats."
        ),
        "metadata": {"strategy": "stem_layer", "context": "same_bpm_chorus"},
    },
    {
        "text": (
            "For same BPM stem layer transitions, beat alignment is critical. "
            "Detect B's first beat onset from the raw waveform, then place B "
            "so that its first beat lands exactly on A's beat grid. Do not "
            "time-stretch either track. Do not fade A's vocals during the "
            "overlay — both play at full volume. B starts from sample 0 with "
            "its full intro (no trimming). The overlay duration equals B's "
            "intro length before vocals start."
        ),
        "metadata": {"strategy": "stem_layer", "context": "same_bpm_alignment"},
    },
]


def seed(kb: TransitionKnowledgeBase) -> None:
    """Upsert all seed rules into the knowledge base (idempotent)."""
    for entry in SEED_RULES:
        kb.add_rule(entry["text"], metadata=entry.get("metadata"))
    logger.info("Knowledge base seeded: %d rules (%d total entries)",
                len(SEED_RULES), kb.count())

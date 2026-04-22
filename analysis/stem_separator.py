"""Stem separation using demucs (Meta's hybrid transformer model).

Splits a track into four stems: vocals, drums, bass, other (melody/synths).
"""

import logging

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from models.track import Stems, Track

logger = logging.getLogger("dj.analysis.stems")

# Lazy-loaded model singleton (htdemucs is the best quality model)
_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading demucs model (htdemucs)...")
        _model = get_model("htdemucs")
        _model.eval()
        if torch.cuda.is_available():
            _model.cuda()
            logger.info("Using CUDA for stem separation")
        else:
            logger.info("Using CPU for stem separation (this may be slow)")
    return _model


def separate_stems(track: Track) -> Track:
    """Run demucs stem separation and store results on the Track."""
    if track.audio is None:
        raise ValueError("Track audio must be loaded before stem separation")

    model = _get_model()

    # demucs expects (channels, samples) tensor — our audio is mono
    audio_tensor = torch.from_numpy(track.audio).float().unsqueeze(0)  # (1, samples)
    # Model expects stereo — duplicate mono to stereo
    audio_tensor = audio_tensor.repeat(2, 1)  # (2, samples)
    # Add batch dimension: (1, 2, samples)
    audio_tensor = audio_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()

    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=audio_tensor.device)

    # sources shape: (1, n_sources, 2, samples) — take mean across stereo channels
    sources = sources.squeeze(0).cpu().numpy()  # (n_sources, 2, samples)

    # Map model source names to our Stems fields
    # htdemucs order: drums, bass, other, vocals
    source_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    stem_map = {}
    for i, name in enumerate(source_names):
        # Average stereo to mono
        stem_map[name] = sources[i].mean(axis=0).astype(np.float32)

    track.stems = Stems(
        vocals=stem_map.get("vocals"),
        drums=stem_map.get("drums"),
        bass=stem_map.get("bass"),
        other=stem_map.get("other"),
    )

    logger.info("Separated stems for %s: %s", track.title, list(stem_map.keys()))
    return track

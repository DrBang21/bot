"""Song structure analysis — detects intro, verse, chorus, drop, bridge, outro.

Primary approach: when beat data is available, compute energy per bar and
detect boundaries where energy changes significantly. This is naturally
bar-aligned and much more precise than spectral novelty for DJ music.

Fallback: recurrence-matrix novelty detection when no beat data exists.

Classification uses a Self-Similarity Matrix (SSM) to identify repeated
sections regardless of their energy level.  Choruses repeat; verses don't.
This handles tracks where the chorus is quieter than the verse (energy
inversion), which breaks energy-only classifiers.
"""

import logging

import librosa
import numpy as np

from models.track import Section, Track

logger = logging.getLogger("dj.analysis.structure")

# Minimum section length in seconds (avoids micro-segments).
# Real song sections are typically 12-30s.  Anything shorter gets
# merged with its neighbor during boundary detection or post-processing.
MIN_SECTION_SEC = 10.0

# Energy thresholds are computed relative to the track's own section energies.
# Slightly wider percentiles reduce false "chorus"/"bridge" labels on songs
# whose RMS values are tightly clustered.
HIGH_ENERGY_PERCENTILE = 0.80
LOW_ENERGY_PERCENTILE = 0.25

# SSM: cosine similarity threshold for declaring two sections "the same"
# (i.e. a repeated chorus).  0.87 keeps verse-vs-verse false positives low
# while still catching chorus instances that share the same hook/beat.
SSM_SIMILARITY_THRESHOLD = 0.93
# Minimum section-index separation to count a pair as a non-adjacent repeat.
SSM_MIN_SEPARATION = 2
# Threshold for merging adjacent raw segments that are nearly identical.
# A false boundary inside a section (e.g. a brief energy dip inside a chorus)
# splits one section into two very similar halves — merge them.
ADJ_MERGE_THRESHOLD = 0.96


def detect_structure(track: Track) -> Track:
    """Detect song sections and store them on the Track."""
    if track.audio is None:
        raise ValueError("Track audio must be loaded before structure analysis")
    if len(track.audio) == 0:
        track.sections = []
        return track

    sr = track.sample_rate
    y = track.audio
    hop_length = 512
    duration = len(y) / sr

    # --- Compute audio features ---
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    S_low = S[:20, :]  # Lowest 20 mel bands (~0-500Hz)
    low_energy = np.mean(S_low, axis=0)

    # Normalize to 0-1 range
    rms_norm = rms / (np.max(rms) + 1e-8)
    centroid_norm = centroid / (np.max(centroid) + 1e-8)
    low_norm = low_energy / (np.max(low_energy) + 1e-8)

    # --- Compute harmonic features (reused for both boundaries and SSM) ---
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    chroma_feat = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    feat_times = librosa.frames_to_time(
        np.arange(mfcc_feat.shape[1]), sr=sr, hop_length=hop_length
    )

    # --- Detect boundaries ---
    if track.downbeats is not None and len(track.downbeats) >= 8:
        boundaries = _boundaries_from_bar_energy(
            track.downbeats, rms_norm, rms_times, duration,
            mfcc=mfcc_feat, chroma=chroma_feat, feat_times=feat_times,
        )
        logger.debug("Bar-energy boundaries: %s",
                      [f"{b:.1f}" for b in boundaries])
    else:
        boundaries = _boundaries_from_novelty(y, sr, hop_length, track, duration)

    # --- Build sections from boundaries ---
    raw_segments: list[dict] = []

    for i in range(len(boundaries) - 1):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])

        if end - start < MIN_SECTION_SEC * 0.5:
            if raw_segments:
                raw_segments[-1]["end"] = end
            continue

        mask = (rms_times >= start) & (rms_times < end)
        seg_energy = float(np.mean(rms_norm[mask])) if np.any(mask) else 0.0
        seg_brightness = float(np.mean(centroid_norm[mask])) if np.any(mask) else 0.0
        seg_bass = float(np.mean(low_norm[mask])) if np.any(mask) else 0.0

        raw_segments.append({
            "idx": i, "start": start, "end": end,
            "energy": seg_energy, "brightness": seg_brightness, "bass": seg_bass,
        })

    if not raw_segments:
        track.sections = []
        return track

    # --- SSM: find repeated sections (chorus candidates) ---
    seg_features = _segment_feature_vectors(raw_segments, mfcc_feat, chroma_feat, feat_times)

    # Merge adjacent raw segments that are nearly identical — these are
    # sections split by a false boundary (e.g. an energy dip inside a chorus).
    raw_segments, seg_features = _merge_adjacent_similar(
        raw_segments, seg_features, threshold=ADJ_MERGE_THRESHOLD
    )

    logger.info(
        "Segments after adjacent merge: %s",
        [f"{s['start']:.0f}-{s['end']:.0f}s" for s in raw_segments],
    )

    repeated_indices, sim_pairs = _ssm_repeated_segments(seg_features, return_pairs=True)
    has_ssm_chorus = len(repeated_indices) > 0
    if sim_pairs:
        pair_str = ", ".join(
            f"seg{i}↔seg{j}({v:.2f})" for i, j, v in sim_pairs
        )
        logger.info("SSM pairs (threshold=%.2f): %s → repeated segs %s",
                    SSM_SIMILARITY_THRESHOLD, pair_str, sorted(repeated_indices))
    else:
        logger.info("SSM: no repeated sections found (threshold=%.2f) → energy fallback",
                    SSM_SIMILARITY_THRESHOLD)

    # --- Classify each section ---
    energies = [s["energy"] for s in raw_segments]
    energy_high = float(np.percentile(energies, HIGH_ENERGY_PERCENTILE * 100))
    energy_low = float(np.percentile(energies, LOW_ENERGY_PERCENTILE * 100))
    # If energy spread is tiny, percentile cutoffs become unstable and cause
    # oscillating labels. Clamp to a minimum separation around the median.
    if (energy_high - energy_low) < 0.08:
        center = float(np.median(energies))
        energy_high = min(1.0, center + 0.04)
        energy_low = max(0.0, center - 0.04)

    sections: list[Section] = []
    n_segs = len(raw_segments)

    for i, seg in enumerate(raw_segments):
        prev_e = raw_segments[i - 1]["energy"] if i > 0 else None
        next_e = raw_segments[i + 1]["energy"] if i < n_segs - 1 else None
        label = _classify_section(
            i, n_segs, seg["start"], seg["end"], duration,
            seg["energy"], seg["brightness"], seg["bass"],
            energy_high, energy_low,
            prev_energy=prev_e, next_energy=next_e,
            is_repeated=(i in repeated_indices),
            has_ssm_chorus=has_ssm_chorus,
        )
        sections.append(Section(
            label=label, start=seg["start"], end=seg["end"], energy=seg["energy"],
        ))

    # Store pre-merge sections for lyric refinement (finer resolution)
    track._pre_merge_sections = list(sections)
    sections = _merge_consecutive(sections)
    track.sections = sections

    labels_summary = " | ".join(
        f"{s.label}({s.start:.0f}-{s.end:.0f}s, E={s.energy:.2f})"
        for s in sections
    )
    logger.info("Structure for %s: %s", track.title, labels_summary)

    return track


# ---------------------------------------------------------------------------
# SSM helpers: feature extraction and repeat detection
# ---------------------------------------------------------------------------

def _segment_feature_vectors(
    segments: list[dict],
    mfcc: np.ndarray,    # (n_mfcc, T)
    chroma: np.ndarray,  # (12, T)
    times: np.ndarray,   # (T,)
) -> np.ndarray:
    """Return a feature vector per segment for SSM classification.

    Uses chroma (12 dims) + low MFCCs 1-5 (5 dims).  Higher MFCCs capture
    fine-grained percussive timbre of the beat loop, which is identical
    across sections in many hip-hop tracks and would cause verse-vs-chorus
    false matches.  Chroma + low MFCCs focus on HARMONIC content where
    chorus hooks differ most from verse rapping.

    Returns shape (n_segs, 17) after cross-segment z-score normalisation.
    """
    N_MFCC_LOW = 5  # MFCCs 1-5 (skip 0 which tracks loudness)

    # Song-level mean for background subtraction.
    # Every track shares a common spectral fingerprint (key, instrumentation).
    # Subtracting the song mean leaves only the *deviation* unique to each
    # section: a chorus hook that emphasises specific pitch classes will have
    # a positive deviation in those chroma bins; a verse with different rap
    # cadence will deviate in different bins.  Two chorus instances then share
    # the same positive/negative deviations → high cosine similarity.
    song_mfcc_mean = np.mean(mfcc[1:N_MFCC_LOW + 1, :], axis=1)   # (5,)
    song_chroma_mean = np.mean(chroma, axis=1)                       # (12,)

    vecs = []
    for seg in segments:
        mask = (times >= seg["start"]) & (times < seg["end"])
        if np.any(mask):
            m = np.mean(mfcc[1:N_MFCC_LOW + 1, mask], axis=1) - song_mfcc_mean
            c = np.mean(chroma[:, mask], axis=1) - song_chroma_mean
        else:
            m = np.zeros(N_MFCC_LOW)
            c = np.zeros(chroma.shape[0])
        vecs.append(np.concatenate([m, c]))

    # L2-normalisation happens inside _ssm_repeated_segments / _merge_adjacent.
    return np.array(vecs, dtype=np.float32)  # (n_segs, 17)


def _merge_adjacent_similar(
    segments: list[dict],
    features: np.ndarray,
    threshold: float = ADJ_MERGE_THRESHOLD,
) -> tuple[list[dict], np.ndarray]:
    """Merge pairs of adjacent segments whose feature vectors are nearly identical.

    A false boundary inside a section (e.g. a brief energy dip mid-chorus)
    splits it into two halves that sound the same.  Merging them gives SSM
    a complete, contiguous section to compare against other instances.

    Only merges one pass (no cascading) to avoid swallowing distinct sections.
    """
    if len(segments) < 2:
        return segments, features

    norms = np.linalg.norm(features, axis=1)

    merged_segs: list[dict] = []
    merged_feats: list[np.ndarray] = []
    skip_next = False

    for i in range(len(segments)):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(segments):
            n1, n2 = norms[i], norms[i + 1]
            if n1 > 1e-8 and n2 > 1e-8:
                adj_sim = float(np.dot(features[i], features[i + 1]) / (n1 * n2))
            else:
                adj_sim = 0.0

            if adj_sim >= threshold:
                s, t = segments[i], segments[i + 1]
                dur_s = s["end"] - s["start"]
                dur_t = t["end"] - t["start"]
                total = max(dur_s + dur_t, 1e-6)
                merged_segs.append({
                    "idx": s["idx"],
                    "start": s["start"],
                    "end": t["end"],
                    "energy": (s["energy"] * dur_s + t["energy"] * dur_t) / total,
                    "brightness": (s["brightness"] * dur_s + t["brightness"] * dur_t) / total,
                    "bass": (s["bass"] * dur_s + t["bass"] * dur_t) / total,
                })
                merged_feats.append((features[i] + features[i + 1]) / 2.0)
                logger.debug(
                    "Adjacent merge: %.0f-%.0fs ↔ %.0f-%.0fs (sim=%.3f)",
                    s["start"], s["end"], t["start"], t["end"], adj_sim,
                )
                skip_next = True
                continue

        merged_segs.append(segments[i])
        merged_feats.append(features[i])

    return merged_segs, np.array(merged_feats, dtype=np.float32)


def _ssm_repeated_segments(
    features: np.ndarray,
    min_sep: int = SSM_MIN_SEPARATION,
    threshold: float = SSM_SIMILARITY_THRESHOLD,
    return_pairs: bool = False,
) -> tuple[set[int], list[tuple[int, int, float]]] | set[int]:
    """Return segment indices that are highly similar to at least one other
    non-adjacent segment.  These are chorus candidates — sections that repeat.

    Uses cosine similarity after L2-normalising each feature vector.

    If return_pairs=True, also returns the list of (i, j, similarity) pairs
    that met the threshold (useful for logging / debugging).
    """
    n = len(features)
    empty: set[int] = set()
    if n < min_sep + 1:
        return (empty, []) if return_pairs else empty

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    feat_norm = features / norms

    sim = (feat_norm @ feat_norm.T).clip(-1.0, 1.0)  # (n, n) cosine similarity

    # Log ALL above-0.70 pairs at debug level for tuning
    for i in range(n):
        for j in range(i + min_sep, n):
            if sim[i, j] >= 0.70:
                logger.debug("SSM similarity seg%d↔seg%d = %.3f", i, j, float(sim[i, j]))

    repeated: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + min_sep, n):
            if sim[i, j] >= threshold:
                repeated.add(i)
                repeated.add(j)
                pairs.append((i, j, float(sim[i, j])))

    return (repeated, pairs) if return_pairs else repeated


# ---------------------------------------------------------------------------
# Boundary detection: bar-grid energy approach (primary)
# ---------------------------------------------------------------------------

def _boundaries_from_bar_energy(
    downbeats: np.ndarray,
    rms_norm: np.ndarray,
    rms_times: np.ndarray,
    duration: float,
    mfcc: np.ndarray | None = None,       # (n_mfcc, T) optional
    chroma: np.ndarray | None = None,     # (12, T) optional
    feat_times: np.ndarray | None = None, # (T,) optional, shared with mfcc/chroma
) -> np.ndarray:
    """Detect section boundaries from energy + harmonic changes at bar boundaries.

    Primary signal: energy change (avg W bars before vs after).
    Secondary signal: harmonic/timbral novelty from MFCC+chroma features.
    Combining both catches transitions that are harmonically distinct even
    when overall energy barely changes (e.g. bridge → final chorus).
    """
    n_db = len(downbeats)

    # Energy per bar
    bar_energies = np.zeros(n_db, dtype=np.float32)
    for i in range(n_db):
        end = downbeats[i + 1] if i + 1 < n_db else duration
        mask = (rms_times >= downbeats[i]) & (rms_times < end)
        if np.any(mask):
            bar_energies[i] = float(np.mean(rms_norm[mask]))

    # MFCC+chroma feature vector per bar (for harmonic novelty)
    has_feat = (mfcc is not None and chroma is not None and feat_times is not None)
    n_feat = (mfcc.shape[0] if mfcc is not None else 0) + (chroma.shape[0] if chroma is not None else 0)
    bar_features = np.zeros((n_db, n_feat), dtype=np.float32)
    if has_feat:
        for i in range(n_db):
            end = downbeats[i + 1] if i + 1 < n_db else duration
            mask = (feat_times >= downbeats[i]) & (feat_times < end)
            if np.any(mask):
                m = np.mean(mfcc[:, mask], axis=1)
                c = np.mean(chroma[:, mask], axis=1)
                bar_features[i] = np.concatenate([m, c])

    # Compute energy change at each bar: avg of W bars before vs W bars after
    W = 2
    change = np.zeros(n_db, dtype=np.float32)
    harmonic_change = np.zeros(n_db, dtype=np.float32)
    for i in range(W, n_db - W):
        before_e = np.mean(bar_energies[i - W:i])
        after_e = np.mean(bar_energies[i:i + W])
        change[i] = abs(after_e - before_e)

        if has_feat:
            before_f = np.mean(bar_features[i - W:i], axis=0)
            after_f = np.mean(bar_features[i:i + W], axis=0)
            nb = np.linalg.norm(before_f)
            na = np.linalg.norm(after_f)
            if nb > 1e-8 and na > 1e-8:
                cos_sim = float(np.dot(before_f, after_f) / (nb * na))
                harmonic_change[i] = max(0.0, 1.0 - cos_sim)

    # Combine energy and harmonic novelty.
    # Normalise each to [0, 1] so scales are comparable, then blend.
    # Energy dominates (0.55) so low-energy transitions with strong
    # harmonic shifts (0.45) still register — e.g. bridge → final chorus.
    max_e = float(np.max(change)) if np.max(change) > 1e-8 else 1.0
    max_h = float(np.max(harmonic_change)) if np.max(harmonic_change) > 1e-8 else 1.0
    combined = (change / max_e) * 0.55 + (harmonic_change / max_h) * 0.45

    # Adaptive threshold per track so quiet and loud songs are treated fairly.
    valid = combined[W:n_db - W]
    nonzero = valid[valid > 0.0]
    if nonzero.size > 0:
        adaptive_thr = float(np.percentile(nonzero, 65)) * 0.75
    else:
        adaptive_thr = 0.0
    threshold = float(np.clip(max(0.015, adaptive_thr), 0.015, 0.08))

    # Minimum section length from bar duration to avoid micro-segments.
    # 6-bar minimum: real section changes in 4/4 hip-hop tracks are typically
    # 8+ bars; 6 bars is a safe lower bound that suppresses internal false
    # boundaries (e.g. a brief energy dip or harmonic wobble mid-chorus)
    # while keeping genuinely close boundaries that differ by exactly 6 bars.
    bar_dur = float(downbeats[1] - downbeats[0]) if n_db > 1 else 2.0
    min_gap_bars = max(6, int(MIN_SECTION_SEC / max(bar_dur, 1e-6)))

    # Collect local maxima once, then apply threshold.
    # Use combined (energy + harmonic) signal for scoring and thresholding.
    all_candidates: list[tuple[int, float]] = []
    candidates: list[tuple[int, float]] = []
    for i in range(W, n_db - W):
        boundary_time = float(downbeats[i])
        # Ignore cuts too close to the start/end of the song.
        if boundary_time < MIN_SECTION_SEC or (duration - boundary_time) < MIN_SECTION_SEC:
            continue
        # Must be >= both immediate neighbors in the combined signal.
        if i > 0 and combined[i] < combined[i - 1]:
            continue
        if i < n_db - 1 and combined[i] < combined[i + 1]:
            continue
        score = float(combined[i])
        if score <= 0.0:
            continue
        all_candidates.append((i, score))
        if score >= threshold:
            candidates.append((i, score))

    # Pick strongest first, enforce minimum gap.
    candidates.sort(key=lambda x: x[1], reverse=True)
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_bars: set[int] = set()
    for idx, _ in candidates:
        if any(abs(idx - s) < min_gap_bars for s in selected_bars):
            continue
        selected_bars.add(idx)

    # Ensure at least a small number of internal cuts for longer tracks.
    min_internal = 2 if duration >= 75.0 else (1 if duration >= 35.0 else 0)
    if len(selected_bars) < min_internal:
        for idx, _ in all_candidates:
            if idx in selected_bars:
                continue
            if any(abs(idx - s) < min_gap_bars for s in selected_bars):
                continue
            selected_bars.add(idx)
            if len(selected_bars) >= min_internal:
                break

    boundaries = [0.0]
    for idx in sorted(selected_bars):
        if idx < n_db:
            boundaries.append(float(downbeats[idx]))
    boundaries.append(duration)

    return np.unique(np.array(boundaries))


# ---------------------------------------------------------------------------
# Boundary detection: novelty fallback (no beat data)
# ---------------------------------------------------------------------------

def _boundaries_from_novelty(
    y: np.ndarray, sr: int, hop_length: int,
    track: Track, duration: float,
) -> np.ndarray:
    """Fallback boundary detection using recurrence-matrix novelty."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    features = np.vstack([
        librosa.util.normalize(chroma),
        librosa.util.normalize(mfcc),
    ])

    rec = librosa.segment.recurrence_matrix(
        features, width=3, mode="affinity", sym=True
    )

    novelty_fn = getattr(librosa.segment, "novelty", None)
    if novelty_fn is not None:
        novelty = novelty_fn(rec)
    else:
        novelty = _novelty_from_recurrence(rec)

    min_section_frames = int(MIN_SECTION_SEC * sr / hop_length)
    boundaries_frames = librosa.util.peak_pick(
        novelty,
        pre_max=min_section_frames,
        post_max=min_section_frames,
        pre_avg=min_section_frames,
        post_avg=min_section_frames,
        delta=0.05,
        wait=min_section_frames,
    )

    boundaries_sec = librosa.frames_to_time(
        boundaries_frames, sr=sr, hop_length=hop_length
    )

    boundaries = np.concatenate([[0.0], boundaries_sec, [duration]])
    return np.unique(boundaries)


def _novelty_from_recurrence(rec: np.ndarray) -> np.ndarray:
    """Approximate novelty when librosa.segment.novelty is unavailable."""
    if rec.ndim != 2 or rec.shape[1] == 0:
        return np.zeros(1, dtype=np.float32)

    diff = np.abs(np.diff(rec, axis=1))
    novelty = np.mean(diff, axis=0)
    novelty = np.concatenate([[0.0], novelty]).astype(np.float32)

    max_val = float(np.max(novelty))
    if max_val > 0.0:
        novelty /= max_val

    return novelty


# ---------------------------------------------------------------------------
# Section classification
# ---------------------------------------------------------------------------

def _classify_section(
    idx: int, n_segments: int,
    start: float, end: float, duration: float,
    energy: float, brightness: float, bass: float,
    energy_high: float, energy_low: float,
    prev_energy: float | None = None,
    next_energy: float | None = None,
    is_repeated: bool = False,
    has_ssm_chorus: bool = False,
) -> str:
    """Classify a section from SSM repetition, position, and energy.

    SSM takes highest priority: a section that repeats non-adjacently is a
    chorus (or drop) regardless of its energy level.  This handles tracks
    where the chorus has *lower* energy than the verse (energy inversion).

    When SSM found at least one repeated pair, the energy-based chorus
    fallback is disabled so that high-energy verses are not mislabelled.

    Tracks can legitimately omit parts (e.g. no bridge or no intro), so
    this logic avoids forcing a fixed structural template.
    """
    rel_start = start / max(duration, 1e-6)
    rel_end = end / max(duration, 1e-6)
    spread = max(energy_high - energy_low, 0.06)
    jump = 0.15 * spread

    # --- SSM-based chorus/drop (highest priority) ---
    # Repeated sections cannot be intro or outro by definition.
    if is_repeated:
        if bass >= 0.55 and bass >= brightness * 1.05:
            return "drop"
        return "chorus"

    # --- Position-based intro ---
    # First segment gets "intro" unless it starts at peak energy
    # (some tracks open directly on the hook).
    if idx == 0 and energy <= energy_high:
        return "intro"

    # --- Position-based outro ---
    if idx == n_segments - 1 and energy <= energy_low + 0.10 * spread:
        return "outro"
    if rel_end >= 0.92 and energy <= energy_low + 0.15 * spread:
        return "outro"

    # --- Early low-energy = intro material ---
    if rel_end <= 0.20 and energy <= energy_low:
        return "intro"

    # --- Bridge: back-half valley after a high-energy section ---
    follows_peak = prev_energy is not None and prev_energy >= energy_high
    next_rises_hard = next_energy is not None and (next_energy - energy) >= jump
    # Use half the track's own spread as the required drop — this scales correctly
    # for tightly-clustered tracks (spread ≈ 0.08) where a fixed 0.05 would be
    # too strict after energy_high has been clamped upward.
    bridge_drop = 0.5 * spread
    if rel_start >= 0.50 and follows_peak and energy <= energy_high - bridge_drop:
        if not next_rises_hard:
            return "bridge"

    # Deep late-track valleys are also bridge-like.
    if rel_start >= 0.65 and energy <= energy_low:
        return "bridge"

    # --- Energy-based chorus (fallback only when SSM found no repeats) ---
    # Disabled when has_ssm_chorus=True to prevent high-energy verses from
    # being mislabelled as chorus on energy-inverted tracks.
    if not has_ssm_chorus and energy >= energy_high:
        if bass >= 0.55 and bass >= brightness * 1.05:
            return "drop"
        return "chorus"

    return "verse"


def refine_sections_with_lyrics(track: Track) -> None:
    """Public entry point: refine an already-detected structure using lyrics.

    Uses pre-merge sections (finer resolution) when available, then re-merges.
    """
    if not track.sections or track.lyrics is None or len(track.lyrics.words) == 0:
        return
    # Use pre-merge sections for finer resolution if available
    source = track._pre_merge_sections if track._pre_merge_sections else track.sections
    track.sections = _refine_with_lyrics(source, track.lyrics)
    labels_summary = " | ".join(
        f"{s.label}({s.start:.0f}-{s.end:.0f}s, E={s.energy:.2f})"
        for s in track.sections
    )
    logger.info("Structure (lyric-refined) for %s: %s", track.title, labels_summary)


def _refine_with_lyrics(sections: list[Section], lyrics) -> list[Section]:
    """Re-label sections using lyric repetition.

    Choruses repeat the same words each time they appear.  Extract the
    text of each section from the word-level transcription, then compare
    all pairs.  Sections whose text overlaps significantly are choruses.
    """
    if len(sections) < 3:
        return sections

    # Extract ordered word list per section
    words = lyrics.words
    section_texts: list[list[str]] = []
    for sec in sections:
        sec_words = [
            w.text.lower().strip(",.!?'\"")
            for w in words
            if w.start >= sec.start and w.end <= sec.end + 0.5
            and len(w.text.strip()) > 1  # Skip single-char noise
        ]
        section_texts.append(sec_words)

    # Compare sections using bigram overlap (primary) with unigram fallback.
    # Bigrams capture word ORDER but Whisper is non-deterministic — the same
    # chorus can come back with slightly different transcriptions across runs.
    # The unigram fallback catches cases where word order varies but the same
    # vocabulary appears (e.g. "just keep watching" vs "keep just watching").
    n = len(sections)
    chorus_indices: set[int] = set()
    MIN_WORDS = 6             # Lowered: short choruses still deserve matching
    BIGRAM_THRESHOLD = 0.35   # 35% shared bigrams (Whisper variance tolerance)
    MIN_SHARED_BIGRAMS = 4    # Lowered for shorter sections
    UNIGRAM_THRESHOLD = 0.50  # Fallback: 50% of smaller set shared
    UNIGRAM_BOTH_THRESHOLD = 0.30  # Also require 30% of *larger* set shared
    MIN_SHARED_UNIGRAMS = 5   # At least 5 shared words

    # Common filler words that Whisper loves to hallucinate or vary
    _STOPWORDS = {
        "the", "a", "an", "and", "but", "or", "in", "on", "at", "to",
        "is", "it", "of", "i", "my", "me", "we", "oh", "ah", "uh",
        "ya", "na", "la", "da", "mm", "hm",
    }

    section_bigrams: list[set[tuple[str, str]]] = []
    section_unigrams: list[set[str]] = []
    for text in section_texts:
        filtered = [w for w in text if w not in _STOPWORDS and len(w) > 1]
        if len(filtered) >= MIN_WORDS:
            bigrams = set(zip(filtered[:-1], filtered[1:]))
            unigrams = set(filtered)
        else:
            bigrams = set()
            unigrams = set(filtered) if len(filtered) >= 3 else set()
        section_bigrams.append(bigrams)
        section_unigrams.append(unigrams)

    for i in range(n):
        for j in range(i + 2, n):  # Skip adjacent
            matched = False

            # Primary: bigram matching
            if section_bigrams[i] and section_bigrams[j]:
                shared = len(section_bigrams[i] & section_bigrams[j])
                smaller = min(len(section_bigrams[i]), len(section_bigrams[j]))
                ratio = shared / smaller if smaller > 0 else 0.0
                if ratio >= BIGRAM_THRESHOLD and shared >= MIN_SHARED_BIGRAMS:
                    matched = True
                    logger.debug("Lyric bigram match: sections %d↔%d share "
                                  "%d/%d bigrams (%.0f%%)",
                                  i, j, shared, smaller, ratio * 100)

            # Fallback: unigram matching (handles Whisper word-order variance)
            # Requires overlap against BOTH the smaller and larger sets to
            # prevent false matches where a short verse shares a few common
            # words with a long chorus.
            if not matched and section_unigrams[i] and section_unigrams[j]:
                shared_u = len(section_unigrams[i] & section_unigrams[j])
                smaller_u = min(len(section_unigrams[i]), len(section_unigrams[j]))
                larger_u = max(len(section_unigrams[i]), len(section_unigrams[j]))
                ratio_small = shared_u / smaller_u if smaller_u > 0 else 0.0
                ratio_large = shared_u / larger_u if larger_u > 0 else 0.0
                if (ratio_small >= UNIGRAM_THRESHOLD
                        and ratio_large >= UNIGRAM_BOTH_THRESHOLD
                        and shared_u >= MIN_SHARED_UNIGRAMS):
                    matched = True
                    logger.debug("Lyric unigram match: sections %d↔%d share "
                                  "%d words (%.0f%% of smaller, %.0f%% of larger)",
                                  i, j, shared_u, ratio_small * 100, ratio_large * 100)

            if matched:
                chorus_indices.add(i)
                chorus_indices.add(j)

    if not chorus_indices:
        return sections

    # Re-label matched sections as "chorus", and DOWNGRADE sections that
    # were labeled "chorus" by SSM but whose lyrics don't match any
    # lyric-identified chorus (they were mislabeled).
    updated = []
    for i, sec in enumerate(sections):
        if i in chorus_indices and sec.label not in ("intro", "outro"):
            updated.append(Section(
                label="chorus", start=sec.start, end=sec.end, energy=sec.energy,
            ))
        elif sec.label == "chorus" and i not in chorus_indices:
            # SSM labeled this "chorus" but lyrics disagree — downgrade to verse
            logger.info("Lyric refinement: downgrading section %d (%s, %.0f-%.0fs) "
                        "from chorus to verse (lyrics don't match)",
                        i, sec.label, sec.start, sec.end)
            updated.append(Section(
                label="verse", start=sec.start, end=sec.end, energy=sec.energy,
            ))
        else:
            updated.append(sec)

    # Re-merge in case relabeling created consecutive same-label sections
    updated = _merge_consecutive(updated)

    logger.info("Lyric refinement: sections %s marked as chorus",
                 sorted(chorus_indices))
    return updated


def _merge_consecutive(sections: list[Section]) -> list[Section]:
    """Merge consecutive sections with the same label."""
    if not sections:
        return sections

    merged = [sections[0]]
    for s in sections[1:]:
        if s.label == merged[-1].label:
            prev = merged[-1]
            prev_dur = prev.end - prev.start
            s_dur = s.end - s.start
            total_dur = prev_dur + s_dur
            if total_dur <= 0:
                avg_energy = (prev.energy + s.energy) / 2.0
            else:
                avg_energy = (prev.energy * prev_dur + s.energy * s_dur) / total_dur
            merged[-1] = Section(
                label=prev.label, start=prev.start, end=s.end, energy=avg_energy,
            )
        else:
            merged.append(s)

    return merged

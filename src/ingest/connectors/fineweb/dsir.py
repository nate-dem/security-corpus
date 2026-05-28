"""
DSIR (Data Selection via Importance Resampling) core implementation.

Reference: Xie et al., "Data Selection for Language Models via Importance Resampling"
           NeurIPS 2023. https://arxiv.org/abs/2302.03169

The model:
    Documents are represented as bag-of-n-grams (unigrams + bigrams) hashed into
    m buckets. Each bucket j has a frequency γ[j]. The log importance weight of a
    document x is:

        log w(x) = Σ_j count(x, j) × log(γ_target[j] / γ_raw[j])

    Documents with high log w(x) look more like the target distribution than the
    raw corpus. We threshold on log w to select a cyber-relevant subset of FineWeb.

Classes:
    DSIRFitter   — fits γ from a stream of texts (call once for target, once for raw)
    DSIRScorer   — combines two fitted γ arrays into a fast per-document scorer

Usage (see scripts/fit_dsir.py for the full workflow):
    fitter_target = DSIRFitter().fit(target_texts)
    fitter_raw    = DSIRFitter().fit(raw_sample_texts)
    scorer        = DSIRScorer.from_fitters(fitter_target, fitter_raw)
    score         = scorer.score(document_text)    # float in [-clip, +clip]
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_M: int = 100_000         # hash bucket count (DSIR paper recommendation for technical domains)
DEFAULT_SMOOTHING: float = 1e-5  # Laplace smoothing to avoid log(0) for unseen n-grams
DEFAULT_CLIP: float = 10.0       # log-weight clipping bounds: [-clip, +clip]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash_ngram(ngram: str, m: int) -> int:
    """Map an n-gram string to a bucket index in [0, m) via CRC32 (deterministic)."""
    return zlib.crc32(ngram.encode()) % m


def _extract_ngram_indices(text: str, m: int) -> list[int]:
    """
    Tokenize (lowercase whitespace split) and return bucket indices for all
    unigrams and bigrams in text.
    """
    tokens = text.lower().split()
    indices = [_hash_ngram(t, m) for t in tokens]
    indices += [_hash_ngram(tokens[i] + " " + tokens[i + 1], m) for i in range(len(tokens) - 1)]
    return indices


# ---------------------------------------------------------------------------
# DSIRFitter
# ---------------------------------------------------------------------------

class DSIRFitter:
    """
    Fits a bag-of-n-grams distribution γ over m hash buckets.

    Call fit() once on a corpus (target or raw sample). The resulting
    gamma_ attribute is a probability vector of shape (m,).

    Serializes to/from pickle via save() / load().
    """

    def __init__(self, m: int = DEFAULT_M, smoothing: float = DEFAULT_SMOOTHING):
        self.m = m
        self.smoothing = smoothing
        self.gamma_: np.ndarray | None = None  # shape (m,), set after fit()
        self._n_docs: int = 0
        self._n_ngrams: int = 0

    def fit(self, texts: Iterable[str]) -> "DSIRFitter":
        """
        Accumulate n-gram bucket counts over texts, then normalize with
        Laplace smoothing to produce self.gamma_.
        """
        counts = np.zeros(self.m, dtype=np.float64)

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            indices = _extract_ngram_indices(text, self.m)
            if not indices:
                continue
            np.add.at(counts, indices, 1)
            self._n_docs += 1
            self._n_ngrams += len(indices)

        # Laplace smoothing: add ε to every bucket before normalizing
        counts += self.smoothing
        self.gamma_ = counts / counts.sum()
        return self

    def save(self, path: Path | str) -> None:
        import pickle
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "DSIRFitter":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        fitted = self.gamma_ is not None
        return (
            f"DSIRFitter(m={self.m}, smoothing={self.smoothing}, "
            f"fitted={fitted}, n_docs={self._n_docs:,}, n_ngrams={self._n_ngrams:,})"
        )


# ---------------------------------------------------------------------------
# DSIRScorer
# ---------------------------------------------------------------------------

class DSIRScorer:
    """
    Scores documents by log importance weight relative to a target distribution.

        log w(x) = Σ_j count(x, j) × log(γ_target[j] / γ_raw[j])

    Scores are clipped to [-clip, +clip] to prevent extreme weights from
    dominating resampling.

    Serializes to/from pickle via save() / load().
    """

    def __init__(
        self,
        gamma_target: np.ndarray,
        gamma_raw: np.ndarray,
        clip: float = DEFAULT_CLIP,
    ):
        if gamma_target.shape != gamma_raw.shape:
            raise ValueError(
                f"gamma shape mismatch: target {gamma_target.shape} vs raw {gamma_raw.shape}"
            )
        self.m = len(gamma_target)
        self.clip = clip
        # Precompute log ratio once; reused for every scored document
        self.log_ratio_: np.ndarray = np.log(gamma_target) - np.log(gamma_raw)  # shape (m,)

    @classmethod
    def from_fitters(
        cls,
        target_fitter: DSIRFitter,
        raw_fitter: DSIRFitter,
        clip: float = DEFAULT_CLIP,
    ) -> "DSIRScorer":
        """Construct a scorer from two fitted DSIRFitter instances."""
        if target_fitter.gamma_ is None or raw_fitter.gamma_ is None:
            raise ValueError("Both fitters must be fitted before creating a scorer.")
        if target_fitter.m != raw_fitter.m:
            raise ValueError(
                f"m mismatch: target m={target_fitter.m}, raw m={raw_fitter.m}"
            )
        return cls(target_fitter.gamma_, raw_fitter.gamma_, clip)

    def score(self, text: str) -> float:
        """
        Return the per-n-gram average log importance weight for a document.

        We normalize by the number of n-grams rather than using the raw sum.
        The raw sum grows linearly with document length, which causes all long
        FineWeb documents (~1,100 n-grams) to clip at the floor even when they
        contain cyber-relevant content — because common English n-grams
        accumulate more negative mass than cyber n-grams can offset. Dividing
        by n-gram count makes scores comparable across document lengths.
        """
        if not text or not text.strip():
            return float(-self.clip)
        indices = _extract_ngram_indices(text, self.m)
        if not indices:
            return float(-self.clip)
        s = float(self.log_ratio_[indices].sum()) / len(indices)
        return float(np.clip(s, -self.clip, self.clip))

    def score_batch(self, texts: list[str]) -> np.ndarray:
        """Score a list of texts. Returns float32 array of shape (n,)."""
        return np.array([self.score(t) for t in texts], dtype=np.float32)

    def save(self, path: Path | str) -> None:
        import pickle
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "DSIRScorer":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"DSIRScorer(m={self.m}, clip={self.clip})"


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(p || q) = Σ_j p[j] log(p[j] / q[j]).

    Both arrays must be valid probability distributions (positive, sum to 1).
    """
    return float(np.sum(p * (np.log(p) - np.log(q))))


def kl_reduction(
    gamma_target: np.ndarray,
    gamma_raw: np.ndarray,
    gamma_selected: np.ndarray,
) -> float:
    """
    KL reduction = KL(target || raw) - KL(target || selected).

    Measures how much closer to the target the selected subset is vs.
    random selection from the raw corpus. Higher is better; positive means
    selection improved alignment.
    """
    return kl_divergence(gamma_target, gamma_raw) - kl_divergence(gamma_target, gamma_selected)


def gumbel_topk_indices(
    log_weights: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Select k indices proportional to exp(log_weights) without replacement,
    using the Gumbel-max trick (Vieira 2014):

        perturbed[i] = log_weights[i] + Gumbel(0, 1)
        top-k = argsort(perturbed, descending)[:k]

    Returns a sorted index array of shape (k,).
    """
    if rng is None:
        rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(log_weights))
    perturbed = log_weights + gumbel_noise
    return np.sort(np.argpartition(perturbed, -k)[-k:])

"""Small text-only CPU classifiers; scores are not calibrated quality guarantees."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from scripts.youtube.profile import _sha256

FEATURES = dict(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    lowercase=True,
)
TARGETS = {
    "domain": ("security_relevance", ("central", "supporting")),
    "substance": ("technical_substance", ("substantive",)),
    "usability": ("text_usability", ("usable",)),
}


def vectorizer():
    return HashingVectorizer(**FEATURES)


def load_model(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "model.json").read_text())
    if manifest["runtime_sha256"] != _sha256(Path(__file__)):
        raise ValueError(
            "Model runtime changed; preserve this model and rebuild separately"
        )
    if manifest["weights_sha256"] != _sha256(folder / "weights.npz"):
        raise ValueError("Model weights checksum mismatch")
    with np.load(folder / "weights.npz", allow_pickle=False) as weights:
        coef, intercept = weights["coef"], weights["intercept"]
    if coef.shape != (len(TARGETS), FEATURES["n_features"]) or intercept.shape != (
        len(TARGETS),
    ):
        raise ValueError("Invalid model dimensions")
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError("Invalid model weights")
    return manifest, coef, intercept


def predict(texts, model):
    _, coef, intercept = model
    logits = vectorizer().transform(texts) @ coef.T + intercept
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

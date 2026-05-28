#!/usr/bin/env python3
"""
Fit DSIR models and run threshold calibration.

Reads the target corpus (built by build_dsir_target.py), samples raw docs
from FineWeb parquets, fits DSIRFitter on both, builds a DSIRScorer, then
evaluates KL-reduction at a range of score thresholds to guide selection
in the ingestion pipeline.

Outputs (all in --output-dir, default data/dsir/):
    fitter_target.pkl       DSIRFitter fit on CTIBench + structured sources
    fitter_raw.pkl          DSIRFitter fit on raw FineWeb sample
    scorer.pkl              DSIRScorer (precomputed log_ratio_ = log(γ_t/γ_r))
    calibration.json        Per-threshold KL reduction + recommended threshold

Usage:
    # Pilot (uses all locally available FineWeb parquets, ~4,860 docs)
    python3 scripts/fit_dsir.py

    # Full run on Marlowe (after full FineWeb ingest, 1M doc sample)
    python3 scripts/fit_dsir.py --sample-size 1000000

    # Point at a specific FineWeb dump directory
    python3 scripts/fit_dsir.py --raw-parquet-dir data/fineweb/normalized/confidence=background
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Warn if raw sample is below this — calibration percentiles will be unreliable.
_MIN_SAMPLE_FOR_CALIBRATION = 50_000

# Percentile thresholds to evaluate during calibration.
_CALIBRATION_PERCENTILES = [10, 25, 50, 60, 70, 75, 80, 85, 90, 95, 99]

# Minimum retention fraction a threshold must keep to be considered for recommendation.
_MIN_RETENTION = 0.01


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_target_texts(path: Path) -> list[str]:
    """Load texts from data/dsir/target_texts.parquet."""
    if not path.exists():
        raise FileNotFoundError(
            f"Target corpus not found: {path}\n"
            "Run scripts/build_dsir_target.py first."
        )
    table = pq.read_table(str(path))
    texts = table["text"].to_pylist()
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    logger.info(f"Target corpus: {len(texts):,} texts from {path}")
    return texts


def _read_fineweb_parquet(path: Path) -> list[str]:
    """Read content column from a single FineWeb normalized parquet."""
    try:
        table = pq.ParquetFile(str(path)).read()
    except Exception as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return []
    if "content" not in table.column_names:
        logger.warning(f"{path} has no 'content' column; available: {table.column_names}")
        return []
    return [t for t in table["content"].to_pylist() if isinstance(t, str) and t.strip()]


def sample_raw_texts(raw_dir: Path, sample_size: int, seed: int = 42) -> list[str]:
    """
    Collect up to sample_size texts from FineWeb normalized parquets.

    Globs recursively for *.parquet under raw_dir so it works whether
    raw_dir points at confidence=background/ or the parent normalized/ dir.
    """
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No *.parquet files found under {raw_dir}\n"
            "Run scripts/ingest_fineweb.py to build the local raw corpus."
        )
    logger.info(f"Raw source: {len(parquet_files)} parquet file(s) under {raw_dir}")

    all_texts: list[str] = []
    for f in parquet_files:
        texts = _read_fineweb_parquet(f)
        all_texts.extend(texts)
        logger.info(f"  {f.parent.name}/{f.name}: {len(texts):,} rows (running total: {len(all_texts):,})")
        if len(all_texts) >= sample_size:
            break

    if len(all_texts) > sample_size:
        rng = random.Random(seed)
        all_texts = rng.sample(all_texts, sample_size)
        logger.info(f"Sampled down to {sample_size:,} docs")

    if len(all_texts) < _MIN_SAMPLE_FOR_CALIBRATION:
        logger.warning(
            f"Raw sample is only {len(all_texts):,} docs "
            f"(recommended ≥ {_MIN_SAMPLE_FOR_CALIBRATION:,} for reliable calibration). "
            "Run the full FineWeb ingest on Marlowe for production use."
        )

    logger.info(f"Raw sample: {len(all_texts):,} texts total")
    return all_texts


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_calibration(
    scores: np.ndarray,
    raw_texts: list[str],
    target_gamma: np.ndarray,
    raw_gamma: np.ndarray,
    m: int,
    smoothing: float,
) -> dict:
    """
    Evaluate KL reduction at each percentile threshold.

    For each threshold t:
      - Select docs with score > t  (retention subset)
      - Fit γ_selected on those docs
      - KL reduction = KL(γ_target || γ_raw) - KL(γ_target || γ_selected)

    Returns a dict suitable for JSON serialization.
    """
    from ingest.connectors.fineweb.dsir import DSIRFitter, kl_divergence, kl_reduction

    kl_baseline = kl_divergence(target_gamma, raw_gamma)
    logger.info(f"KL(target || raw) baseline: {kl_baseline:.4f}")

    score_percentiles = {
        f"p{p}": float(np.percentile(scores, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }
    logger.info(f"Score distribution: {score_percentiles}")

    threshold_results = []
    for pct in _CALIBRATION_PERCENTILES:
        t = float(np.percentile(scores, pct))
        selected_mask = scores > t
        n_selected = int(selected_mask.sum())
        retention = n_selected / len(scores)

        if n_selected < 2:
            continue

        selected_texts = [raw_texts[i] for i, keep in enumerate(selected_mask) if keep]
        fitter_sel = DSIRFitter(m=m, smoothing=smoothing).fit(selected_texts)
        kl_red = kl_reduction(target_gamma, raw_gamma, fitter_sel.gamma_)

        threshold_results.append({
            "percentile": pct,
            "threshold": round(t, 4),
            "n_selected": n_selected,
            "retention_rate": round(retention, 4),
            "kl_reduction": round(kl_red, 4),
        })
        logger.info(
            f"  p{pct:2d}  threshold={t:6.2f}  "
            f"retention={retention:.1%}  "
            f"kl_reduction={kl_red:.4f}"
        )

    # Recommend: highest KL reduction subject to retention >= _MIN_RETENTION
    eligible = [r for r in threshold_results if r["retention_rate"] >= _MIN_RETENTION]
    recommended = max(eligible, key=lambda r: r["kl_reduction"]) if eligible else None
    if recommended:
        logger.info(
            f"\nRecommended threshold: {recommended['threshold']} "
            f"(p{recommended['percentile']}, "
            f"retention={recommended['retention_rate']:.1%}, "
            f"kl_reduction={recommended['kl_reduction']:.4f})"
        )
    else:
        logger.warning("No eligible threshold found — all candidates drop below min retention.")

    return {
        "n_target": None,           # filled in by caller
        "n_raw_sample": len(scores),
        "kl_baseline": round(kl_baseline, 4),
        "score_percentiles": score_percentiles,
        "thresholds": threshold_results,
        "recommended": recommended,
        "settings": {
            "m": m,
            "smoothing": smoothing,
            "min_retention": _MIN_RETENTION,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit DSIR models and calibrate selection threshold."
    )
    parser.add_argument(
        "--target", default="data/dsir/target_texts.parquet",
        help="Target corpus parquet (default: data/dsir/target_texts.parquet)",
    )
    parser.add_argument(
        "--raw-parquet-dir", default="data/fineweb/normalized",
        help=(
            "Directory of FineWeb normalized parquets to use as raw sample. "
            "Globs recursively for *.parquet. "
            "(default: data/fineweb/normalized)"
        ),
    )
    parser.add_argument(
        "--sample-size", type=int, default=1_000_000,
        help="Max raw docs to sample for fitting γ_raw (default: 1,000,000)",
    )
    parser.add_argument(
        "--output-dir", default="data/dsir",
        help="Directory for output artifacts (default: data/dsir)",
    )
    parser.add_argument(
        "--m", type=int, default=100_000,
        help="Number of hash buckets (default: 100,000)",
    )
    parser.add_argument(
        "--smoothing", type=float, default=1e-5,
        help="Laplace smoothing epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--clip", type=float, default=10.0,
        help="Log-weight clipping bound (default: 10.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for raw sampling (default: 42)",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Fit and save models only; skip KL calibration pass.",
    )
    args = parser.parse_args()

    from ingest.connectors.fineweb.dsir import DSIRFitter, DSIRScorer

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load corpora ---
    target_texts = load_target_texts(repo_root / args.target)
    raw_texts = sample_raw_texts(repo_root / args.raw_parquet_dir, args.sample_size, args.seed)

    # --- Fit fitters ---
    logger.info(f"\nFitting γ_target on {len(target_texts):,} target texts (m={args.m}) ...")
    fitter_target = DSIRFitter(m=args.m, smoothing=args.smoothing).fit(target_texts)
    logger.info(f"  {fitter_target}")

    logger.info(f"Fitting γ_raw on {len(raw_texts):,} raw texts (m={args.m}) ...")
    fitter_raw = DSIRFitter(m=args.m, smoothing=args.smoothing).fit(raw_texts)
    logger.info(f"  {fitter_raw}")

    # --- Build scorer ---
    scorer = DSIRScorer.from_fitters(fitter_target, fitter_raw, clip=args.clip)
    logger.info(f"  {scorer}")

    # --- Save artifacts ---
    fitter_target.save(output_dir / "fitter_target.pkl")
    fitter_raw.save(output_dir / "fitter_raw.pkl")
    scorer.save(output_dir / "scorer.pkl")
    logger.info(f"\nSaved fitter_target.pkl, fitter_raw.pkl, scorer.pkl → {output_dir}")

    # --- Calibration ---
    if args.skip_calibration:
        logger.info("Skipping calibration (--skip-calibration).")
        return

    logger.info("\nRunning threshold calibration on raw sample ...")
    scores = scorer.score_batch(raw_texts)

    calibration = run_calibration(
        scores=scores,
        raw_texts=raw_texts,
        target_gamma=fitter_target.gamma_,
        raw_gamma=fitter_raw.gamma_,
        m=args.m,
        smoothing=args.smoothing,
    )
    calibration["n_target"] = len(target_texts)

    cal_path = output_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Calibration written → {cal_path}")


if __name__ == "__main__":
    main()

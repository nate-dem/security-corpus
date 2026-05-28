# CLAUDE.md — security-corpus

Cybersecurity LLM mid-training corpus pipeline. Ingests structured and unstructured security sources, normalizes to a canonical Parquet schema, filters and routes records into RegMix-ready buckets, and optimizes data mixture weights via proxy training + regression.

---

## Current Priorities

1. **FineWeb full Marlowe run** — DSIR pipeline is complete locally. On Marlowe: run `fit_dsir.py --sample-size 1000000` on unfiltered FineWeb to get a real γ_raw, re-calibrate thresholds, then run `ingest_fineweb.py --slurm --tasks 64`.
2. **YouTube transcripts** — replace the current `_passes_gate()` filter in `src/ingest/connectors/youtube_transcripts.py` (curated allowlist + keyword gate, ~20M tokens from 634 channels) with a lightweight lenient metadata classifier.
3. **RegMix full run** — scripts 01–05 exist; `01_prepare_buckets.py` must run on Marlowe after full FineWeb ingest populates `data/fineweb/normalized/confidence=high/`. Token counts in `buckets.yaml` are mostly zero.
4. **Populate remaining structured source buckets** — NVD, Sigma, StackExchange, MITRE, CAPEC, CISA-KEV are not yet ingested. Run `scripts/ingest_*.py` for each on Marlowe.

---

## Data Sources

### Structured (connectors exist)

| Source | `source_id` | Schema | Ingest script |
|---|---|---|---|
| NVD | `nvd` | `VulnerabilityData` | `scripts/ingest_nvd.py` |
| CISA KEV | `cisa-kev` | `VulnerabilityData` | `scripts/ingest_cisa_kev.py` |
| MITRE ATT&CK | `mitre-attack` | `MitreData` | `scripts/ingest_mitre_attack.py` |
| MITRE CWE | `mitre-cwe` | `MitreData` | `scripts/ingest_mitre_cwe.py` |
| CAPEC | `capec` | `MitreData` | `scripts/ingest_capec.py` |
| BRON | `bron` | `NormalizedData` | `scripts/ingest_bron.py` |
| Sigma Rules | `sigma` | `DetectionRuleData` | `scripts/ingest_sigma.py` |
| GitHub Advisory | `github-advisory` | `VulnerabilityData` | `scripts/ingest_github_advisory.py` |
| StackExchange (infosec, reverseengineering, crypto, tor) | `stackexchange-{slug}` | `QAThreadData` | `scripts/ingest_stackexchange.py <site>` |
| YouTube Transcripts | `youtube-transcripts` | `TranscriptData` | `scripts/ingest_youtube_transcripts.py` |

### Web-scale (streaming)

| Source | `source_id` | Schema | Ingest script |
|---|---|---|---|
| FineWeb V1.4.0 | `fineweb` | `FineWebData` | `scripts/ingest_fineweb.py` |

**On disk today**: `bron` (778 records), `youtube-transcripts` (15 records), `fineweb` (DSIR-scored pilot, ~5k docs). Everything else needs ingestion on Marlowe.

**Not yet ingested**: `reddit_cyber` — `BUCKET_SOURCES["reddit_cyber"] = []` in `01_prepare_buckets.py`; no connector.

---

## Pipeline Architecture

### Stage 1 — Ingestion → Normalized Parquet

Each connector implements `iter_records(path)` and `normalize(record) -> NormalizedData`. Entry point: `src/ingest/pipeline.py::ingest_and_store()`. Output:

```
data/{source_id}/normalized/source_id={source_id}/raw.parquet
```

FineWeb is the exception — it writes directly via Datatrove to:
```
data/fineweb/normalized/confidence={high,medium,background}/raw_{dump}.parquet
```

The primary text field across **all** `NormalizedData` subclasses is **`content`**, not `text`.

---

### Stage 2 — FineWeb DSIR Pipeline

FineWeb is too large to ingest normally. It is streamed from HuggingFace via a Datatrove pipeline with DSIR-based importance scoring:

```
ParquetReader (HuggingFace hf://datasets/HuggingFaceFW/fineweb/data)
  → QualityGateFilter   (token_count >= 40, language_score >= 0.65)
  → DSIRScoringStep     (loads scorer.pkl, scores each doc, annotates dsir_score + confidence)
  → TieredParquetWriter (routes to confidence=high / medium / background)
```

**DSIR requires these artifacts to exist first** (`data/dsir/`):

| Artifact | Built by | Description |
|---|---|---|
| `target_texts.parquet` | `build_dsir_target.py` | 5,403 texts from CTIBench + BRON + YouTube |
| `fitter_target.pkl` | `fit_dsir.py` | γ_target: n-gram distribution of target corpus |
| `fitter_raw.pkl` | `fit_dsir.py` | γ_raw: n-gram distribution of raw FineWeb sample |
| `scorer.pkl` | `fit_dsir.py` | DSIRScorer: precomputed log_ratio_ = log(γ_target/γ_raw) |
| `calibration.json` | `fit_dsir.py` | KL reduction at percentile thresholds, recommended threshold |

**DSIR score formula** (per document x):

```
log w(x) = (1 / |ngrams(x)|) × Σ_j count(x,j) × log(γ_target[j] / γ_raw[j])
```

Normalized by n-gram count to remove length bias (FineWeb docs avg ~1,100 n-grams; CTIBench avg ~136). Clipped to [-10, +10].

**Tier routing** (defaults in `ingest_fineweb.py`):
- `dsir_score >= 0.0` → `high`
- `dsir_score >= -0.3` → `medium`
- otherwise → `background`

On Marlowe: re-run `fit_dsir.py --sample-size 1000000` with unfiltered FineWeb to get a real γ_raw, then use `calibration.json` recommended threshold to update `--high-threshold` and `--medium-threshold`.

**DSIR core classes** — `src/ingest/connectors/fineweb/dsir.py`:
- `DSIRFitter(m=100_000, smoothing=1e-5)` — fits γ from texts via `.fit(texts)`
- `DSIRScorer.from_fitters(target_fitter, raw_fitter, clip=10.0)` — precomputes `log_ratio_`
- `kl_divergence(p, q)`, `kl_reduction(target, raw, selected)` — calibration utilities
- `gumbel_topk_indices(log_weights, k)` — Gumbel-max weighted sampling without replacement

---

### Stage 3 — RegMix Bucket Preparation

`regmix/scripts/01_prepare_buckets.py` — tokenizes corpus sources into `data/buckets/{bucket_name}/shard_XXXX.parquet`. Source mapping in `BUCKET_SOURCES` (top of that file). FineWeb entries point to directories and are resolved via `resolve_sources()` which globs `*.parquet`.

**9 RegMix buckets** (`regmix/config/buckets.yaml`):
`mitre_cve`, `sigma_rules`, `bron_graph`, `stackexchange_security`, `youtube_cyber`, `reddit_cyber`, `github_security`, `security_blogs`, `general_technical`

- `security_blogs` ← `data/fineweb/normalized/confidence=high/`
- `general_technical` ← `data/fineweb/normalized/confidence=background/`

---

### Stage 4 — RegMix Pipeline (scripts 02–05)

Run in order after Stage 3:

| Script | Purpose | Key output |
|---|---|---|
| `02_sample_mixtures.py` | Dirichlet + reference mixtures | `experiments/mixtures.jsonl` |
| `03_run_proxy_jobs.py` | Proxy training per mixture | `experiments/results/m????.json` |
| `04_fit_regression.py` | Ridge + LightGBM fit | `experiments/regression/model.pkl` |
| `05_simulate_and_select.py` | 100k simulation + selection | `experiments/selected_mixture.json` |

RegMix samples mixture weights from Dirichlet(α = λ · x₀) where x₀ is the natural token distribution. Proxy models are trained on each mixture, evaluated on CTIBench/AttackQA/FAITH, and the results fed into a regression model. The regression predicts which mixture minimizes `ValidationLosses.composite(lambda_penalty, general_baseline)`.

---

### Stage 5 — Validation

Benchmark targets in `regmix/config/experiment.yaml`: `cyber_general`, `cloud_security`, `task_specific`, `general_language`.

Key benchmarks: **CTIBench** (NeurIPS 2024, 5 tasks, 4,610 samples at `AI4Sec/cti-bench`), **AttackQA**, **MITRE**, **CVE/CWE/CAPEC**, **FAITH**.

Validation logic: `regmix/evaluation/validator.py`.

---

## CTIBench Tasks

Downloaded to `data/ctibench/` by `scripts/download_ctibench.py`. HuggingFace ID: `AI4Sec/cti-bench`.

| File | Task | Rows | Text column |
|---|---|---|---|
| `cti_mcq.parquet` | Multiple-choice CTI knowledge | 2,500 | `Question` |
| `cti_rcm.parquet` | CVE → CWE mapping | 1,000 | `Description` |
| `cti_vsp.parquet` | CVE → CVSS score prediction | 1,000 | `Description` |
| `cti_ate.parquet` | ATT&CK technique extraction | 60 | `Description` |
| `cti_taa.parquet` | Threat actor attribution | 50 | `Text` (no GT — withheld) |

The `Prompt` column in every file contains prompt-formatted text — do NOT use it for DSIR fitting.

---

## Coding Rules

- **Primary text field is `content`**, not `text`. All `NormalizedData` subclasses use `content`. Any script reading normalized parquets must use `TEXT_COLUMN = "content"`.
- **Normalized output layout**: `data/{source_id}/normalized/source_id={source_id}/raw.parquet` — match exactly when adding sources.
- **Datatrove import paths**: `from datatrove.pipeline.filters.base_filter import BaseFilter` (not package-level). `PipelineStep` and `DocumentsPipeline` at `datatrove.pipeline.base`.
- **`Document` dataclass** has only `id`, `media`, `metadata`, `text` — no `score` attribute.
- **Datatrove executor caches** completions in `logs/`. Delete `logs/fineweb/{dump}/` before re-running a dump.
- **Do not name a Counter attribute `stats`** on a Datatrove wrapper class — conflicts with Datatrove's internal `stats.to_dict()` call at teardown.
- **FineWeb multi-dump output**: `raw_{dump}.parquet` per dump when processing multiple; `raw.parquet` for single-dump runs.
- **pyarrow schema merging**: use `pq.ParquetFile(path).read()` instead of `pq.read_table(path)` for Hive-partitioned files. The Hive partition in the path (e.g. `source_id=bron/`) conflicts with the same column inside the file when encodings differ.
- **StackExchange `source_id`**: `stackexchange-{slug}` (hyphen, not underscore). Registered slugs: `infosec`, `reverseengineering`, `crypto`, `tor`.

---

## Known Pitfalls

- **DSIR length normalization is required.** `DSIRScorer.score()` divides the log-weight sum by n-gram count before clipping. Without this, FineWeb docs (~1,100 n-grams avg) accumulate so many negative contributions from common English n-grams that all scores clip to -10.0. CTIBench docs average only ~136 n-grams — the length mismatch makes the unnormalized sum useless.
- **DSIR pilot calibration is biased.** Local `data/fineweb/normalized/` only has the keyword-scored background tier (~4,860 docs). Fitting γ_raw on this produces a biased raw distribution — background is pre-filtered non-cyber. On Marlowe, fit γ_raw on 1M unfiltered docs streamed directly from HuggingFace.
- **`reddit_cyber` bucket is empty**: expected, not a bug. Script warns and skips.
- **YouTube over-filtering**: `_passes_gate()` in `src/ingest/connectors/youtube_transcripts.py` uses a hard channel allowlist + keyword gate. Produced only ~20M tokens from 634 channels. Replacement should be a lenient metadata classifier.
- **FineWeb pilot sequences first row group only**: `--max-docs N` reads the first N docs sequentially from each dump shard. Trusted-domain pages (nvd.nist.gov, cisa.gov) are not in the first row group — retention rates from pilots undercount these.
- **Datatrove inner-class pattern**: `QualityGateFilter`, `DSIRScoringStep`, `TieredParquetWriter` all wrap Datatrove base classes via a `_Inner` subclass. This is required by Datatrove's metaclass system. The outer class handles construction; `_Inner` handles pipeline execution.
- **Old `raw.parquet` files** in `data/fineweb/normalized/` are from the pre-DSIR keyword-scored run and lack the `dsir_score` column. Delete them before running `01_prepare_buckets.py` to avoid schema confusion.

---

## Do-Not-Do Rules

- Do not use `text` as the column name for document content — it is `content` everywhere in NormalizedData subclasses.
- Do not use the unnormalized DSIR score sum — always divide by n-gram count (already done in `DSIRScorer.score()`).
- Do not fit γ_raw on keyword-pre-filtered FineWeb — use unfiltered streaming from HuggingFace on Marlowe.
- Do not propose Ollama / LLM inference for full FineWeb classification — at 7B params, classifying 18.5M candidates costs ~78 GPU-years.
- Do not commit ingested parquet data to git — `data/` is gitignored.
- Do not create new `NormalizedData` subclasses without registering `source_id` in `src/ingest/pipeline.py::_CONNECTORS`.
- Do not add RegMix buckets without updating both `regmix/config/buckets.yaml` and `BUCKET_SOURCES` in `regmix/scripts/01_prepare_buckets.py`.
- Do not skip `resolve_sources()` in `01_prepare_buckets.py` — directory-based FineWeb entries silently produce zero files if passed as file paths.

---

## Verified Commands

```bash
# Install
pip install -e ".[dev]"          # base + tests
pip install -e ".[regmix]"       # + RegMix deps (no training)
pip install -e ".[regmix-train]" # + torch/transformers for proxy training
pip install -e ".[fineweb]"      # + datatrove[io] + huggingface_hub + datasets + numpy

# Tests
pytest                           # unit tests
pytest -m data_quality           # validate ingested Parquet output

# DSIR pipeline (run in order before FineWeb ingestion)
python3 scripts/download_ctibench.py                          # → data/ctibench/*.parquet
python3 scripts/build_dsir_target.py                          # → data/dsir/target_texts.parquet
python3 scripts/fit_dsir.py --sample-size 1000000             # → data/dsir/{fitter_target,fitter_raw,scorer}.pkl + calibration.json
                                                               # (on Marlowe with 1M unfiltered docs)

# FineWeb ingestion (requires data/dsir/scorer.pkl)
python3 scripts/ingest_fineweb.py --max-docs 5000              # pilot: 5k docs, all 6 V1.4 dumps
python3 scripts/ingest_fineweb.py --dump CC-MAIN-2025-26       # single dump, no limit
python3 scripts/ingest_fineweb.py --slurm --tasks 64           # full V1.4 run on Marlowe

# Override DSIR thresholds (after calibration on Marlowe)
python3 scripts/ingest_fineweb.py --high-threshold 0.1 --medium-threshold -0.2

# StackExchange
python3 scripts/ingest_stackexchange.py infosec
python3 scripts/ingest_stackexchange.py reverseengineering

# RegMix pipeline (run in order, after 01 has populated token counts)
python3 -m regmix.scripts.01_prepare_buckets --config regmix/config/buckets.yaml --experiment regmix/config/experiment.yaml --output data/buckets
python3 -m regmix.scripts.02_sample_mixtures --config regmix/config/buckets.yaml --experiment regmix/config/experiment.yaml
python3 -m regmix.scripts.03_run_proxy_jobs  --dry-run   # validate without GPU
python3 -m regmix.scripts.04_fit_regression
python3 -m regmix.scripts.05_simulate_and_select

# Clear Datatrove executor cache for a dump (required before re-running a dump)
rm -rf logs/fineweb/

# Inspect output
python3 -c "
import pyarrow.parquet as pq
from pathlib import Path
for f in sorted(Path('data/fineweb/normalized').rglob('*.parquet')):
    t = pq.ParquetFile(str(f)).read()
    print(f'{f.parent.name}/{f.name}: {len(t)} rows')
"
```

---

## DSIR Artifacts Layout

```
data/dsir/
├── target_texts.parquet    5,403 rows, single 'text' column
│                           (4,610 CTIBench + 778 BRON + 15 YouTube)
├── fitter_target.pkl       DSIRFitter(m=100k, n_docs=5403, n_ngrams=733k)
├── fitter_raw.pkl          DSIRFitter(m=100k, n_docs=4860, n_ngrams=5.6M) ← pilot only
├── scorer.pkl              DSIRScorer(m=100k, clip=10.0) — load this in ingest_fineweb.py
└── calibration.json        threshold table + recommended threshold
                            (pilot: recommended=-1.29, p50 — not usable for production)

data/ctibench/
├── cti_mcq.parquet         2,500 rows
├── cti_rcm.parquet         1,000 rows
├── cti_vsp.parquet         1,000 rows
├── cti_ate.parquet            60 rows
└── cti_taa.parquet            50 rows
```

---

## Experiments Directory Layout

```
experiments/
├── fineweb_experiments/          # FineWeb-specific experiment reports
│   └── 01_pilot_v14_keyword_scoring_5k_docs_results.tex
├── results/                      # ProxyRunResult JSON files (m0000–m0031, ref_*)
├── regression/                   # model.pkl, metrics.json, importances.json, cv_metrics.json
├── mixtures.jsonl
├── selected_mixture.json
├── simulation_stats.json
└── top_k_summary.json
```

FineWeb experiment reports go in `experiments/fineweb_experiments/` with naming convention `NN_<descriptive_name>.tex`.

---

## Recent Progress

- **DSIR pipeline fully implemented**: `download_ctibench.py` → `build_dsir_target.py` → `fit_dsir.py` → `ingest_fineweb.py` (DSIR-scored). Replaces keyword scoring end-to-end.
- **`scoring.py` deleted**: keyword scorer (`compute_cyber_score`, `route_confidence`) removed. `FineWebData.dsir_score` replaces `cyber_score`. `FineWebConnector` now accepts an optional `DSIRScorer` for on-the-fly scoring.
- **DSIR pilot run complete**: 4,998 docs scored across 6 V1.4 dumps. Background mean score = -1.29; 5 medium-tier docs identified (scores -0.18 to -0.29). All scores negative — expected, pilot γ_raw was fit on pre-filtered background tier.
- **Key DSIR fix**: length normalization in `DSIRScorer.score()`. Without dividing by n-gram count, all FineWeb docs clipped to -10.0 due to length mismatch between FineWeb (~1,100 n-grams) and CTIBench (~136 n-grams).
- **Next**: full Marlowe FineWeb ingest with `--sample-size 1000000` for production γ_raw; YouTube classifier replacement; populate NVD/Sigma/StackExchange buckets.

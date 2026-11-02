"""Distill existing web span labels; never reuse the previous strict keep votes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from ingest.utils import compute_content_hash
from scripts.web_semantic import model
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256


def examples(root):
    rows, bindings, seen = [], {}, set()
    for path in sorted((root / "packets").glob("part-*.json")):
        packet = json.loads(path.read_text())
        cases = {
            c["text_sha256"]: c
            for c in packet["cases"]
            if c["source"] in ("primus-fineweb", "redsage-cfw")
        }
        if not cases:
            continue
        config_path = root / "runs" / path.stem / "run-config.json"
        config = json.loads(config_path.read_text())
        if config["dry_run"] or config["packet_sha256"] != packet["packet_sha256"]:
            raise ValueError("Teacher configuration/packet mismatch")
        bindings[str(path)] = _sha256(path)
        bindings[str(config_path)] = _sha256(config_path)
        for decision_path in sorted((config_path.parent / "decisions").glob("*.json")):
            decision = json.loads(decision_path.read_text())
            case = cases.get(decision["content_hash"])
            if case is None:
                raise ValueError("Teacher decision has no source document")
            if decision["parse_status"] != "ok":
                continue
            if any(
                decision[k] != config[k]
                for k in ("model", "model_revision", "prompt_version")
            ) or decision["run_config_sha256"] != _sha256(config_path):
                raise ValueError("Teacher model/configuration mismatch")
            text = case["text"]
            start, end = decision["start"], decision["end"]
            if compute_content_hash(text) != case[
                "text_sha256"
            ] or not 0 <= start < end <= len(text):
                raise ValueError("Teacher content hash/span mismatch")
            key = (case["text_sha256"], start, end)
            if key in seen:
                raise ValueError("Duplicate training span")
            seen.add(key)
            labels = decision["labels"]
            if any(
                labels.get(field) == "uncertain" for field, _ in model.TARGETS.values()
            ):
                continue
            bindings[str(decision_path)] = _sha256(decision_path)
            rows.append(
                {
                    "text": text[start:end],
                    "hash": case["text_sha256"],
                    "source": case["source"],
                    "start": start,
                    "end": end,
                    "lineage": case["lineage"],
                    "teacher": {
                        k: config[k]
                        for k in ("model", "model_revision", "prompt_version")
                    },
                    "decision_sha256": _sha256(decision_path),
                    "labels": [
                        int(labels[field] in positive)
                        for field, positive in model.TARGETS.values()
                    ],
                }
            )
    if not rows:
        raise ValueError("No validated training examples")
    return rows, bindings


def fit(rows):
    matrix = model.vectorizer().transform([r["text"] for r in rows])
    labels = np.asarray([r["labels"] for r in rows])
    classifiers = []
    # Training hyperparameters, not document inclusion thresholds. Fixed rather
    # than optimized against the independent token sample.
    for column in range(len(model.TARGETS)):
        learner = LogisticRegression(
            C=10, solver="liblinear", random_state=0, max_iter=500
        )
        learner.fit(matrix, labels[:, column])
        classifiers.append(learner)
    return classifiers


def train(root, output):
    if output.exists():
        raise ValueError("Use a new model output directory")
    start = time.monotonic()
    rows, bindings = examples(root)
    # Whole documents are grouped together. This is teacher-agreement testing,
    # not a domain-disjoint or independently labeled accuracy estimate.
    train_rows = [r for r in rows if int(r["hash"][:8], 16) % 5]
    heldout = [r for r in rows if not int(r["hash"][:8], 16) % 5]
    learners = fit(train_rows)
    matrix = model.vectorizer().transform([r["text"] for r in heldout])
    target = np.asarray([r["labels"] for r in heldout])
    evaluation = {}
    for i, (name, learner) in enumerate(zip(model.TARGETS, learners)):
        evaluation[name] = {
            "classification": classification_report(
                target[:, i], learner.predict(matrix), output_dict=True, zero_division=0
            ),
            "roc_auc": roc_auc_score(target[:, i], learner.predict_proba(matrix)[:, 1]),
        }
    learners = fit(rows)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "weights.npz",
        coef=np.vstack([m.coef_[0] for m in learners]),
        intercept=np.array([m.intercept_[0] for m in learners]),
    )
    _write_json(output / "training-inputs.json", bindings)
    _write_json(output / "training-identities.json", sorted({r["hash"] for r in rows}))
    # Reproduction locators and labels without redistributing source text.
    with (output / "training-examples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {k: v for k, v in row.items() if k != "text"}, sort_keys=True
                )
                + "\n"
            )
    manifest = {
        "version": "web-semantic-linear-v1",
        "runtime_sha256": _sha256(Path(model.__file__)),
        "trainer_sha256": _sha256(Path(__file__)),
        "weights_sha256": _sha256(output / "weights.npz"),
        "training_inputs_sha256": _sha256(output / "training-inputs.json"),
        "training_identities_sha256": _sha256(output / "training-identities.json"),
        "training_examples_sha256": _sha256(output / "training-examples.jsonl"),
        "sklearn_version": sklearn.__version__,
        "features": model.FEATURES,
        "targets": model.TARGETS,
        "spans": len(rows),
        "documents": len({r["hash"] for r in rows}),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "heldout_spans": len(heldout),
        "heldout_teacher_agreement": evaluation,
        "seconds": time.monotonic() - start,
        "release_ready": False,
        "limitations": "Weak supervision from existing Qwen first-pass span labels, two web shards. Document-grouped holdout measures teacher agreement, not truth or unseen-domain generalization. Scores are not calibrated probabilities of correctness. No source/domain names are supplied as features. All source text is vectorized without truncation. No final cutoff is encoded.",
    }
    _write_json(output / "model.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(train(args.teacher.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()

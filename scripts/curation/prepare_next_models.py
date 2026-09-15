"""Download only the two pinned comparison models, on Marlowe scratch storage."""

import json
from pathlib import Path
import platform

from huggingface_hub import snapshot_download

from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter.score import load_tokenizer, _versions
from .first_batch import HERE
from . import rubric_v4


def main():
    if platform.system() != "Linux":
        raise RuntimeError(
            "Run on Marlowe; model weights must not be downloaded to the Mac"
        )
    models = json.loads((HERE / "next_models.json").read_text())["models"]
    for model in models:
        # Validate the small tokenizer/template download before fetching weights.
        tokenizer, _ = load_tokenizer(model["model"], model["revision"], False)
        probe = "A process uses virtual addresses that the operating system maps to physical memory."
        for kind in ("web", "youtube"):
            spans = rubric_v4.make_spans(probe, tokenizer, kind)
            if (
                not spans
                or "".join(probe[s["start"] : s["end"]] for s in spans) != probe
            ):
                raise ValueError("Tokenizer/chat template preflight failed")
        print(
            f"Downloading pinned {model['model']} ({model['weight_bytes'] / 1e9:.2f} GB weights)",
            flush=True,
        )
        snapshot = Path(
            snapshot_download(
                model["model"],
                revision=model["revision"],
                max_workers=4,
                allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"],
            )
        )
        if not list(snapshot.glob("*.safetensors")):
            raise ValueError("Model snapshot contains no weights")
        print(f"Ready: {model['model']} at {snapshot}", flush=True)
    # Snapshot download performs Hub integrity verification; no duplicate 68GB hash pass.
    report = {
        "complete": True,
        "models": models,
        "versions": _versions(False),
        "models_file_sha256": _sha256(HERE / "next_models.json"),
        "requirements_sha256": _sha256(HERE / "requirements-next.txt"),
    }
    _write_json(HERE / ".venv-next" / "models-ready.json", report)


if __name__ == "__main__":
    main()

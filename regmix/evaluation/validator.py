from __future__ import annotations

"""
Validation loss computation for proxy and target runs.

`compute_validation_losses` accepts an already-loaded model and tokenizer
so it can be called inline after proxy training without reloading from disk.

`evaluate_from_checkpoint` loads a saved checkpoint for post-hoc evaluation.
"""

import math
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_validation_losses(
    model,
    tokenizer,
    validation_paths: dict[str, Path],
    device: str = "cpu",
    max_length: int = 2048,
    max_batches: Optional[int] = 200,
    batch_size: int = 4,
) -> dict[str, float]:
    """
    Compute per-token cross-entropy loss on each validation set.

    Args:
        model:              loaded causal LM (already on `device`)
        tokenizer:          corresponding tokenizer
        validation_paths:   {val_name: Path}  to directories of parquet files
        device:             "cuda" or "cpu"
        max_length:         truncate sequences to this length
        max_batches:        stop after this many batches per val set (speed control)
        batch_size:         sequences per forward pass

    Returns:
        {val_name: perplexity-normalized cross-entropy (nats)}
    """
    import torch
    import pyarrow.parquet as pq

    model.eval()
    losses: dict[str, float] = {}

    with torch.no_grad():
        for val_name, val_path in validation_paths.items():
            val_path = Path(val_path)
            files = sorted(val_path.glob("*.parquet"))
            if not files:
                logger.warning(f"No parquet files found in {val_path}, skipping {val_name}")
                continue

            total_loss = 0.0
            total_tokens = 0
            batches_seen = 0

            for fpath in files:
                if max_batches and batches_seen >= max_batches:
                    break
                table = pq.read_table(fpath, columns=["input_ids"])
                rows = table["input_ids"].to_pylist()

                for i in range(0, len(rows), batch_size):
                    if max_batches and batches_seen >= max_batches:
                        break
                    chunk = [r[:max_length] for r in rows[i : i + batch_size]]
                    padded = tokenizer.pad(
                        {"input_ids": chunk},
                        padding=True,
                        return_tensors="pt",
                    )
                    input_ids = padded["input_ids"].to(device)
                    attention_mask = padded["attention_mask"].to(device)
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100  # ignore padding

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    n_tokens = int((labels != -100).sum())
                    total_loss += outputs.loss.item() * n_tokens
                    total_tokens += n_tokens
                    batches_seen += 1

            if total_tokens > 0:
                losses[val_name] = total_loss / total_tokens
            else:
                losses[val_name] = float("inf")
                logger.warning(f"No tokens evaluated for {val_name}")

    model.train()
    return losses


def evaluate_from_checkpoint(
    checkpoint_path: str | Path,
    validation_paths: dict[str, Path],
    device: Optional[str] = None,
    max_length: int = 2048,
    max_batches: Optional[int] = 200,
) -> dict[str, float]:
    """Load a saved checkpoint and compute validation losses."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("Requires torch and transformers.") from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path), torch_dtype=torch.bfloat16
    ).to(device)

    return compute_validation_losses(
        model=model,
        tokenizer=tokenizer,
        validation_paths=validation_paths,
        device=device,
        max_length=max_length,
        max_batches=max_batches,
    )


def measure_general_baseline(
    base_model: str,
    general_validation_path: Path,
    device: Optional[str] = None,
    max_batches: int = 100,
) -> float:
    """
    Measure the base model's general language loss before any midtraining.
    Store this value in experiment.yaml as `objective.general_baseline`.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("Requires torch and transformers.") from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to(device)

    losses = compute_validation_losses(
        model=model,
        tokenizer=tokenizer,
        validation_paths={"general_language": general_validation_path},
        device=device,
        max_batches=max_batches,
    )
    return losses["general_language"]

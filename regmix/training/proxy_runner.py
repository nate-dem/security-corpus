from __future__ import annotations

"""
Proxy midtraining runner.

Each call to `run_proxy_job` starts from the same frozen base checkpoint,
trains for a fixed token budget on a specified mixture, and returns the
validation losses needed for regression.

The implementation wraps HuggingFace Transformers + Accelerate and is
designed to run on a single GPU (PoC) or be launched via accelerate launch
for multi-GPU proxy runs.

External dependencies (not in pyproject.toml yet):
    torch, transformers, accelerate, datasets
"""

import copy
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_proxy_job(
    run_id: str,
    mixture_weights: dict[str, float],
    bucket_paths: dict[str, Path],
    validation_paths: dict[str, Path],
    base_model: str,
    token_budget: int,
    training_cfg: dict,
    output_dir: Optional[Path] = None,
    keep_checkpoint: bool = False,
) -> dict[str, float]:
    """
    Run one proxy midtraining job and return validation losses.

    Args:
        run_id:             unique identifier for this run
        mixture_weights:    {bucket_name: weight}  (must sum to 1)
        bucket_paths:       {bucket_name: Path}  to tokenized parquet dirs
        validation_paths:   {val_set_name: Path}  to tokenized parquet dirs
        base_model:         HuggingFace model id or local path
        token_budget:       total training tokens for this proxy run
        training_cfg:       dict matching experiment.yaml training section
        output_dir:         where to write the checkpoint (tmp dir if None)
        keep_checkpoint:    delete checkpoint after eval if False

    Returns:
        {val_set_name: cross-entropy loss}
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_cosine_schedule_with_warmup,
        )
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "Proxy training requires torch and transformers. "
            "Install with: pip install torch transformers accelerate"
        ) from e

    from regmix.buckets.dataset import WeightedBucketDataset
    from regmix.evaluation.validator import compute_validation_losses

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = copy.deepcopy(training_cfg)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
    ).to(device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset = WeightedBucketDataset(
        bucket_paths=bucket_paths,
        weights=mixture_weights,
        token_budget=token_budget,
        seed=cfg.get("seed", 42),
    )

    batch_size = cfg.get("batch_size", 32)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    max_length = cfg.get("max_length", 2048)

    def collate(batch):
        ids = [ex["input_ids"][:max_length] for ex in batch]
        padded = tokenizer.pad(
            {"input_ids": ids},
            padding=True,
            return_tensors="pt",
        )
        padded["labels"] = padded["input_ids"].clone()
        return padded

    loader = DataLoader(
        list(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.get("dataloader_num_workers", 2),
    )

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    steps_per_epoch = len(loader)
    total_optimizer_steps = max(1, steps_per_epoch // grad_accum)
    warmup_steps = int(total_optimizer_steps * cfg.get("warmup_ratio", 0.05))

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 0.1),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    step = 0

    logger.info(f"[{run_id}] Starting proxy run: {token_budget:,} tokens, device={device}")

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("gradient_clip", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 100 == 0:
                logger.info(f"[{run_id}] step={step}  loss={accum_loss:.4f}")
            accum_loss = 0.0

    # ------------------------------------------------------------------
    # Save checkpoint (optionally)
    # ------------------------------------------------------------------
    _tmp_dir = None
    if output_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix=f"regmix_{run_id}_")
        output_dir = Path(_tmp_dir)
    else:
        output_dir = Path(output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    val_losses = compute_validation_losses(
        model=model,
        tokenizer=tokenizer,
        validation_paths=validation_paths,
        device=device,
        max_length=max_length,
    )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if not keep_checkpoint:
        import shutil
        shutil.rmtree(str(output_dir), ignore_errors=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[{run_id}] Done. Val losses: {val_losses}")
    return val_losses

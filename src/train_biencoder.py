"""
Training script for the improved bi-encoder.

Usage (from the challenge repo root):
    uv run python /path/to/train_biencoder.py

Or copy this into participant/train.py with modified config.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import logging

from participant.validate import callback as validation_callback
from workrb_challenge.training import (
    DataConfig,
    EarlyStopping,
    LossConfig,
    LossLogger,
    ModelCheckpoint,
    ModelConfig,
    OptimConfig,
    SamplerConfig,
    TargetConfig,
    TrainConfig,
    apply_cli_overrides,
    train,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    config = TrainConfig(
        # Model: Improved bi-encoder with all-mpnet-base-v2
        model=ModelConfig(
            target="src.model_biencoder:ImprovedBiEncoder",
            init={
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "max_length": 192,
                "leaderboard_name": "ImprovedBiEncoder-v1",
                "leaderboard_description": "MPNet-v2 + multi-layer pooling + temp=0.03",
                "temperature": 0.03,
                "encode_batch_size": 256,
            },
        ),

        # Data: Multiple ESCO datasets for better generalization
        data=DataConfig(
            dataset=TargetConfig(
                target="participant.data:SkillSentenceDataset",
                init={
                    "dataset_names": [
                        "TechWolf/Synthetic-ESCO-skill-sentences",
                    ],
                    "split": "train",
                },
            ),
            sampler=SamplerConfig(
                target="participant.sampler:RandomBatchSampler",
                init={"shuffle": True},
            ),
            collate="participant.data:default_collate",
            batch_size=128,  # Larger batch = more negatives
            num_workers=4,
        ),

        # Loss: Temperature-scaled InfoNCE
        loss=LossConfig(
            target="participant.loss:InfoNCELoss",
            init={"temperature": 0.03},
        ),

        # Optimizer: AdamW with cosine schedule
        optim=OptimConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
        ),

        # Schedule
        epochs=3,
        seed=42,
        log_every=50,

        output_dir=None,

        # Callbacks
        callbacks=[
            LossLogger(log_every=50),
            validation_callback,
            ModelCheckpoint(
                every_epochs=1,
                save_last=True,
                monitor="ndcg@100_macro",
                mode="max",
            ),
            EarlyStopping(
                monitor="ndcg@100_macro",
                mode="max",
                patience=3,
                min_delta=0.001,
            ),
        ],
    )

    config = apply_cli_overrides(config)
    train(config)


if __name__ == "__main__":
    main()

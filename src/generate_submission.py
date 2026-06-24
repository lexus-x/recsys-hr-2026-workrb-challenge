"""
Generate submission file for CodaBench.

This script generates the submission JSON and ZIP file.
Copy this to the challenge repo's submission/ directory and configure.

Usage:
    uv run python generate_submission.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import workrb
from workrb.models.base import ModelInterface
from workrb.tasks import (
    ESCOGradedSkillNormRanking,
    HouseGradedSkillExtractRanking,
    SkillSkapeGradedSkillExtractRanking,
    TechGradedSkillExtractRanking,
    TechWolfGradedSkillExtractRanking,
)
from workrb.tasks.abstract.base import DatasetSplit
from workrb.tasks.abstract.ranking_base import RankingTask

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to "test" for final submission, "validation" for local testing
SPLIT = "test"

# Model configuration - point to your trained model
MODEL_DEF_FILE = "participant/my_model.py"  # Or your custom model file
WEIGHTS_PATH = ""  # Path to trained weights, empty for no-weights baseline

OUTPUT_FILE = "submission/submission.json"

# Task definitions
TASKS: list[dict[str, Any]] = [
    {"task": TechGradedSkillExtractRanking, "metrics": ["ndcg@100"], "languages": ["en"]},
    {"task": HouseGradedSkillExtractRanking, "metrics": ["ndcg@100"], "languages": ["en"]},
    {"task": TechWolfGradedSkillExtractRanking, "metrics": ["ndcg@100"], "languages": ["en"]},
    {"task": SkillSkapeGradedSkillExtractRanking, "metrics": ["ndcg@100"], "languages": ["en"]},
    {"task": ESCOGradedSkillNormRanking, "metrics": ["ndcg@100"], "languages": ["en"]},
]

TOP_K: int | None = 500
TARGET_ID_PREFIX = "http://data.europa.eu/esco/skill/"


def main() -> None:
    """Generate submission file."""
    split = DatasetSplit("test" if SPLIT == "test" else "val")

    # Load model
    logger.info("Loading model from %s (weights=%s)", MODEL_DEF_FILE, WEIGHTS_PATH or "<none>")

    if WEIGHTS_PATH:
        from workrb_challenge.models import WorkrbSaveable
        model = WorkrbSaveable.from_pretrained(WEIGHTS_PATH)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("model", MODEL_DEF_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Find the ModelInterface subclass
        candidates = [
            obj for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, ModelInterface) and obj is not ModelInterface
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected 1 ModelInterface subclass, found {len(candidates)}")
        model = candidates[0]()

    logger.info("Model: %s", model.name)

    # Build submission
    submission: dict[str, dict[str, dict[str, Any]]] = {model.name: {}}

    for task_cfg in TASKS:
        task_cls = task_cfg["task"]
        try:
            task = task_cls(split=split.value, languages=task_cfg.get("languages"))
        except Exception as e:
            logger.warning("Skipping %s: %s", task_cls.__name__, e)
            continue

        if not isinstance(task, RankingTask):
            continue

        for dataset_id in task.dataset_ids:
            dataset = task.datasets[dataset_id]
            logger.info("[%s/%s] %d queries x %d targets",
                       task.name, dataset_id, len(dataset.query_texts), len(dataset.target_space))

            start = time.time()
            matrix = task.compute_prediction_matrix(model=model, dataset_id=dataset_id)
            matrix = np.asarray(matrix, dtype=np.float32)
            logger.info("Inference: %.1fs", time.time() - start)

            # Get query/target IDs
            from datasets import load_dataset
            hf_name = getattr(task, "hf_name", None)
            if hf_name:
                split_to_hf = getattr(task, "split_to_hf_split", None)
                queries_hf_split = split_to_hf.get(task.split, "validation") if split_to_hf else "validation"

                queries_df = load_dataset(hf_name, "queries", split=queries_hf_split).to_pandas()
                corpus_df = load_dataset(hf_name, "corpus", split="corpus").to_pandas()

                text_to_qid = {str(t).strip(): str(qid) for qid, t in zip(queries_df["_id"], queries_df["text"])}
                title_to_cid = {str(t).strip(): str(cid) for cid, t in zip(corpus_df["_id"], corpus_df["title"])}

                query_ids = [text_to_qid[q] for q in dataset.query_texts]
                target_ids = [title_to_cid[t] for t in dataset.target_space]
            else:
                query_ids = [str(i) for i in range(len(dataset.query_texts))]
                target_ids = [str(i) for i in range(len(dataset.target_space))]

            # Build score dict
            short_ids = [
                tid[len(TARGET_ID_PREFIX):] if tid.startswith(TARGET_ID_PREFIX) else tid
                for tid in target_ids
            ]

            scores = {}
            for qi in range(matrix.shape[0]):
                row = matrix[qi]
                if TOP_K and TOP_K < len(row):
                    keep = np.argpartition(row, -TOP_K)[-TOP_K:]
                else:
                    keep = range(len(row))
                scores[query_ids[qi]] = {short_ids[ti]: float(row[ti]) for ti in keep}

            submission[model.name][task.name] = {
                dataset_id: {
                    "num_queries": len(dataset.query_texts),
                    "num_targets": len(dataset.target_space),
                    "scores": scores,
                }
            }

    # Write output
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, allow_nan=False)
    logger.info("Wrote: %s", out_path)

    # Write ZIP for CodaBench
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=out_path.name)
    logger.info("Upload this to CodaBench: %s", zip_path)


if __name__ == "__main__":
    main()

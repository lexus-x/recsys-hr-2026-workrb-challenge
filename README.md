# RecSys-HR 2026 WorkRB Challenge

Solution for the [RecSys-HR 2026 WorkRB Challenge](https://www.codabench.org/competitions/16900/) — skill extraction and normalization against the ESCO taxonomy with graded relevance.

## Task

Given free-form text (job descriptions, resumes), extract mentioned skills and normalize them to ESCO concepts (13K+ skills). Evaluated with nDCG@100 on graded relevance labels (0-4).

Five datasets: House, Tech, TechWolf, SkillSkape, SkillNorm.

## Approach

**Bi-encoder retrieval** with `all-mpnet-base-v2`, fine-tuned on ESCO synthetic data with InfoNCE loss (temp=0.03). Optional cross-encoder re-ranking on top-200 candidates.

Key design choices:
- 192 max length (skill descriptions need context)
- Multi-layer pooling (last 4 hidden states)
- Temperature-scaled cosine similarity
- Batch size 128 for more in-batch negatives

## Usage

```bash
# Clone official challenge repo
git clone https://github.com/techwolf-ai/workrb-recsys-hr-challenge-2026.git
cd workrb-recsys-hr-challenge-2026 && uv sync

# Train
uv run python participant/train.py \
    --set model.init.model_name="sentence-transformers/all-mpnet-base-v2" \
    --set model.init.max_length=192 \
    --set data.batch_size=128 \
    --set epochs=3

# Generate submission
uv run python submission/generate_submission_file.py
```

Upload `submission/submission.zip` to CodaBench.

## Files

- `src/model_biencoder.py` — improved bi-encoder
- `src/model_cross_encoder.py` — cross-encoder re-ranker
- `src/train_biencoder.py` — training config
- `src/generate_submission.py` — submission generator
- `configs/best_config.yaml` — hyperparameters
- `GETTING_STARTED.md` — step-by-step guide

# RecSys-HR 2026: WorkRB Challenge Solution

🏆 **Target: Top 3 finish for €2000/€1000/€500 prizes + €1000 student track**

## Challenge Overview

- **Task**: Skill extraction + normalization against ESCO (13,000+ skills)
- **Metric**: nDCG@100 with graded relevance (0-4)
- **Datasets**: House, Tech, TechWolf, SkillSkape, SkillNorm
- **Deadline**: July 31, 2026
- **Submissions**: 10 max during test phase

## Strategy: Multi-Stage Pipeline

### Stage 1: Strong Bi-Encoder Retrieval
- Fine-tune `sentence-transformers/all-mpnet-base-v2` on ESCO synthetic data
- Use hard negative mining with in-batch negatives
- Temperature-scaled InfoNCE loss

### Stage 2: Cross-Encoder Re-Ranking
- Re-rank top-100 candidates per query with a cross-encoder
- Fine-tuned on graded relevance labels (0-4)

### Stage 3: LLM Re-Ranking (Optional)
- Use Qwen2.5-1.5B-Instruct for top-50 re-ranking
- Yes/No logit margin scoring

## Files

```
├── README.md
├── STRATEGY.md              # Detailed strategy notes
├── src/
│   ├── model_biencoder.py   # Fine-tuned bi-encoder
│   ├── model_cross_encoder.py # Cross-encoder re-ranker
│   ├── model_llm_reranker.py  # LLM re-ranker
│   ├── model_ensemble.py    # Combined pipeline
│   ├── train_biencoder.py   # Training script
│   ├── train_cross_encoder.py
│   └── generate_submission.py
├── configs/
│   └── best_config.yaml
└── scripts/
    ├── run_all.sh
    └── evaluate.sh
```

## Quick Start

```bash
# Clone the official challenge repo
git clone https://github.com/techwolf-ai/workrb-recsys-hr-challenge-2026
cd workrb-recsys-hr-challenge-2026
uv sync

# Clone this solution
git clone https://github.com/lexus-x/recsys-hr-2026-workrb-challenge.git
```

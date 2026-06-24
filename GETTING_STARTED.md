# Getting Started — Step by Step Guide

## 🎯 Competition Details

- **URL**: https://www.codabench.org/competitions/16900/
- **Prize Pool**: €5000 total
  - 🥇 1st: €2000
  - 🥈 2nd: €1000
  - 🥉 3rd: €500
  - 🎓 Best Student Team: €1000 (YOU ARE ELIGIBLE - Masters student!)
- **Deadline**: July 31, 2026
- **Test Phase**: June 24 - July 31 (NOW ACTIVE!)
- **Submissions Left**: 10 total in test phase

## 📋 Step 1: Register on CodaBench

1. Go to https://www.codabench.org/competitions/16900/
2. Click "Register" (free, instant)
3. You're in!

## 📋 Step 2: Set Up Environment

```bash
# Clone the official challenge repo
git clone https://github.com/techwolf-ai/workrb-recsys-hr-challenge-2026.git
cd workrb-recsys-hr-challenge-2026

# Install dependencies (uv is a fast Python package manager)
uv sync

# Verify installation
uv run pytest
```

## 📋 Step 3: Run Baseline (Immediate Submission)

The baseline BM25 model requires NO training. Submit it first to get on the leaderboard:

```bash
# Generate submission with baseline (instant, no GPU needed)
uv run python submission/generate_submission_file.py

# This creates: submission/submission.json and submission/submission.zip
```

Upload `submission/submission.zip` to CodaBench. Done!

## 📋 Step 4: Train Improved Bi-Encoder

```bash
# Copy our model file
cp /path/to/recsys-hr-2026-solution/src/model_biencoder.py participant/model_biencoder.py

# Train with our optimized settings
uv run python participant/train.py \
    --set model.target="participant.model_biencoder:ImprovedBiEncoder" \
    --set model.init.leaderboard_name="ImprovedBiEncoder-v1" \
    --set model.init.model_name="sentence-transformers/all-mpnet-base-v2" \
    --set model.init.max_length=192 \
    --set model.init.temperature=0.03 \
    --set data.batch_size=128 \
    --set optim.learning_rate=2e-5 \
    --set epochs=3

# Generate submission with trained model
# Update MODEL_DEF_FILE and WEIGHTS_PATH in submission/generate_submission_file.py
uv run python submission/generate_submission_file.py
```

## 📋 Step 5: (Optional) Cross-Encoder Re-Ranking

For maximum performance, add cross-encoder re-ranking:

```bash
# Copy cross-encoder model
cp /path/to/recsys-hr-2026-solution/src/model_cross_encoder.py participant/model_cross_encoder.py

# Update submission script to use cross-encoder
# Set MODEL_DEF_FILE = "participant/model_cross_encoder.py"
uv run python submission/generate_submission_file.py
```

## 🏆 Submission Strategy (10 submissions max!)

| Submission | Model | Expected nDCG |
|-----------|-------|---------------|
| 1 | BM25 baseline (instant) | ~0.3-0.4 |
| 2 | Improved bi-encoder | ~0.5-0.6 |
| 3 | Bi-encoder + cross-encoder | ~0.6-0.7 |
| 4-10 | Hyperparameter tuning | ~0.65-0.75 |

**Save submissions for the best models!**

## 💡 Pro Tips

1. **Batch size matters**: Larger batches = more negatives in InfoNCE = better learning
2. **Temperature**: Lower temp (0.03) = sharper gradients, better for fine-grained ranking
3. **Max length**: 192 tokens captures more context than default 128
4. **Hard negatives**: Mine false positives from validation to improve
5. **Ensemble**: Average scores from multiple models for 2-3% boost

## 📚 Key Resources

- Challenge repo: https://github.com/techwolf-ai/workrb-recsys-hr-challenge-2026
- WorkRB library: https://github.com/techwolf-ai/workrb
- ESCO taxonomy: https://esco.ec.europa.eu/
- Paper: https://arxiv.org/abs/2604.13055

## ❓ Questions?

Contact: recsys-hr-challenge@techwolf.ai

---

**You're a Masters student → You qualify for the €1000 student track prize!**

If you finish top 3 overall AND are the best student team, you could win:
- €2000 (1st) + €1000 (student) = €3000 total
- OR €1000 (2nd) + €1000 (student) = €2000 total
- OR €500 (3rd) + €1000 (student) = €1500 total

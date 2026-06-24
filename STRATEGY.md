# Winning Strategy — RecSys-HR 2026 WorkRB Challenge

## Key Insights

1. **Graded relevance (0-4) vs binary**: The metric rewards getting the *ordering* right, not just binary yes/no
2. **nDCG@100**: Only top 100 predictions matter per query — focus on precision at the top
3. **5 datasets**: Need to generalize across House, Tech, TechWolf, SkillSkape, SkillNorm
4. **ESCO has 13K+ skills**: Dense retrieval + re-ranking is the only scalable approach

## Approach A: Fine-Tuned Bi-Encoder (Primary)

### Model
- Backbone: `sentence-transformers/all-mpnet-base-v2` (768d, strong baseline)
- Mean pooling, L2-normalized embeddings
- Cosine similarity scoring

### Training Data
- `TechWolf/Synthetic-ESCO-skill-sentences` (138K pairs)
- Augment with ESCO skill descriptions + altLabels
- Hard negative mining from top-k false positives

### Loss
- Temperature-scaled InfoNCE (temp=0.03 for sharper gradients)
- Multi-positive: each sentence can match multiple skills

### Hyperparameters
- LR: 2e-5 with cosine schedule + warmup
- Batch size: 128 (more negatives per batch)
- Epochs: 3-5 with early stopping on nDCG@100
- Max length: 192 (longer than default 128 for skill descriptions)

## Approach B: Cross-Encoder Re-Ranker (Secondary)

### Model
- `cross-encoder/ms-marco-MiniLM-L-6-v2` or fine-tuned `all-MiniLM-L6-v2`
- Input: [CLS] query [SEP] skill [SEP]
- Output: graded relevance score (0-4 regression or 5-class classification)

### Training
- Use the graded relevance labels from validation set
- Augment with hard negatives from bi-encoder top-k

## Approach C: LLM Re-Ranker (Bonus)

- Use Qwen2.5-1.5B-Instruct for top-50 re-ranking
- Prompt: "Does this sentence demonstrate skill X? Yes/No"
- Score: logit(Yes) - logit(No)
- Blend with bi-encoder score (weight=0.7 retriever + 0.3 LLM)

## Submission Strategy

With only 10 test submissions:
1. Submit bi-encoder only (baseline)
2. Submit bi-encoder + cross-encoder
3. Submit full pipeline
4. Tune weights based on validation correlation

## Timeline

- Week 1: Bi-encoder training + validation
- Week 2: Cross-encoder training + ensemble
- Week 3: LLM re-ranking + final tuning
- Week 4: Final submissions

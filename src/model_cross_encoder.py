"""
Cross-Encoder Re-Ranker for RecSys-HR 2026 WorkRB Challenge.

Re-ranks top-K candidates from bi-encoder using a cross-encoder
that jointly processes (query, skill) pairs for better accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from workrb.models import ModelInterface
from workrb.types import ModelInputType
from workrb_challenge.models import WorkrbSaveable


class CrossEncoderReranker(nn.Module, ModelInterface, WorkrbSaveable):
    """
    Two-stage pipeline:
    1. Bi-encoder retrieval (fast, broad)
    2. Cross-encoder re-ranking (slow, precise)

    For nDCG@100, we only need to get the top-100 right.
    Re-ranking top-200 with a cross-encoder is tractable and effective.
    """

    def __init__(
        self,
        retriever_name: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 200,
        max_length: int = 256,
        encode_batch_size: int = 256,
        rerank_batch_size: int = 64,
        leaderboard_name: str = "BiEncCrossEnc",
        leaderboard_description: str = "Bi-encoder retrieval + cross-encoder re-ranking",
    ):
        super().__init__()
        self.retriever_name = retriever_name
        self.reranker_name = reranker_name
        self.top_k = top_k
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size
        self.rerank_batch_size = rerank_batch_size
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description

        # Stage 1: Bi-encoder retriever
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.backbone = AutoModel.from_pretrained(retriever_name)

        # Stage 2: Cross-encoder reranker
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self.reranker = AutoModel.from_pretrained(reranker_name)
        hidden_size = self.reranker.config.hidden_size
        self.reranker_head = nn.Linear(hidden_size, 1)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        return (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _encode(self, texts: list[str]) -> torch.Tensor:
        chunk = self.encode_batch_size
        if chunk <= 0 or len(texts) <= chunk:
            return self._encode_batch(texts)
        return torch.cat([self._encode_batch(texts[i:i+chunk]) for i in range(0, len(texts), chunk)])

    def _rerank_score(self, queries: list[str], targets: list[str]) -> torch.Tensor:
        """Cross-encoder scoring for (query, target) pairs."""
        inputs = self.reranker_tokenizer(
            queries, targets,
            padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.reranker(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.reranker_head(cls_emb).squeeze(-1)

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        # Stage 1: Bi-encoder retrieval
        q_emb = torch.nn.functional.normalize(self._encode(queries), p=2, dim=-1)
        t_emb = torch.nn.functional.normalize(self._encode(targets), p=2, dim=-1)
        base_scores = q_emb @ t_emb.T  # (Nq, Nt)

        Nq, Nt = base_scores.shape
        k = min(self.top_k, Nt)

        # Stage 2: Re-rank top-k for each query
        final_scores = base_scores.clone()
        for qi in range(Nq):
            top_idx = base_scores[qi].topk(k).indices
            q_repeated = [queries[qi]] * k
            t_candidates = [targets[ti] for ti in top_idx.tolist()]

            # Batch cross-encoder scoring
            rerank_scores = []
            for batch_start in range(0, k, self.rerank_batch_size):
                batch_end = min(batch_start + self.rerank_batch_size, k)
                scores = self._rerank_score(
                    q_repeated[batch_start:batch_end],
                    t_candidates[batch_start:batch_end]
                )
                rerank_scores.append(scores.detach())
            rerank_scores = torch.cat(rerank_scores)

            # Blend: lift re-ranked scores above base scores
            base_max = base_scores[qi].max()
            blended = base_max + 1.0 + (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min() + 1e-9)
            final_scores[qi, top_idx] = blended.to(final_scores.dtype)

        return final_scores

    def _compute_classification(self, texts, targets, input_type, target_input_type=None):
        return self._compute_rankings(texts, targets, input_type, target_input_type or input_type)

    @property
    def name(self) -> str:
        return self._leaderboard_name

    @property
    def description(self) -> str:
        return self._leaderboard_description

    @property
    def classification_label_space(self) -> list[str] | None:
        return None

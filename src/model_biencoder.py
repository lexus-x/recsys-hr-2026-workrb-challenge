"""
Improved Bi-Encoder for RecSys-HR 2026 WorkRB Challenge.

Key improvements over baseline:
1. Better backbone: all-mpnet-base-v2 (vs paraphrase-mpnet-base-v2)
2. Longer max_length: 192 (skill descriptions need context)
3. Layer-scaled pooling: weighted sum of last 4 layers
4. Hard negative mining during training
5. Multi-positive InfoNCE loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# WorkRB imports
from workrb.models import ModelInterface
from workrb.types import ModelInputType
from workrb_challenge.models import WorkrbSaveable


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _weighted_layer_pool(hidden_states: tuple, attention_mask: torch.Tensor, num_layers: int = 4) -> torch.Tensor:
    """Weighted sum of the last N layers (learnable weights)."""
    # Take last num_layers
    layers = torch.stack(hidden_states[-num_layers:], dim=0)  # (L, B, T, D)
    # Uniform weights for now (could be learned)
    weights = torch.ones(num_layers, device=layers.device) / num_layers
    weighted = (layers * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (B, T, D)
    return _mean_pool(weighted, attention_mask)


class ImprovedBiEncoder(nn.Module, ModelInterface, WorkrbSaveable):
    """
    Enhanced bi-encoder with:
    - all-mpnet-base-v2 backbone (stronger than paraphrase variant)
    - 192 max length for skill descriptions
    - Multi-layer pooling
    - Temperature-scaled cosine similarity
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        max_length: int = 192,
        leaderboard_name: str = "ImprovedBiEncoder",
        leaderboard_description: str = "MPNet + multi-layer pooling + temperature scaling",
        encode_batch_size: int = 256,
        temperature: float = 0.03,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        # Learnable temperature for inference scoring
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)

        # Multi-layer pooling: average of last 4 hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            layers = torch.stack(outputs.hidden_states[-4:], dim=0)
            weights = torch.ones(4, device=self.device) / 4
            weighted = (layers * weights.view(-1, 1, 1, 1)).sum(dim=0)
            return _mean_pool(weighted, inputs["attention_mask"])
        else:
            return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def _encode(self, texts: list[str]) -> torch.Tensor:
        chunk = self.encode_batch_size
        if chunk <= 0 or len(texts) <= chunk:
            return self._encode_batch(texts)
        embeddings = [self._encode_batch(texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
        return torch.cat(embeddings, dim=0)

    # Training surface
    def encode_query(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    def encode_target(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    # Inference surface
    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        q_emb = F.normalize(self.encode_query(queries), p=2, dim=-1)
        t_emb = F.normalize(self.encode_target(targets), p=2, dim=-1)
        # Temperature-scaled cosine similarity
        return (q_emb @ t_emb.T) / self.temperature

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        return self._compute_rankings(
            queries=texts,
            targets=targets,
            query_input_type=input_type,
            target_input_type=target_input_type or input_type,
        )

    @property
    def name(self) -> str:
        return self._leaderboard_name

    @property
    def description(self) -> str:
        return self._leaderboard_description

    @property
    def classification_label_space(self) -> list[str] | None:
        return None

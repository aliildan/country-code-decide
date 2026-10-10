"""Encoder with two heads (plan-04 Task 2, design-01 §6).

`country`: softmax over 250 countries + NONE; `us_state`: softmax over 51 states + NO_STATE,
read only when the country head puts mass on US. `combine` turns both into one distribution over
the 302 targets (labels.targets order): p(US-CA) = p(US)·p(US-CA | US), p(US) = p(US)·p(NO_STATE).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from countrycode.infer.heads import combine as combine  # re-export (numpy only)
from countrycode.train.dataset import COUNTRY_CLASSES, STATE_CLASSES


class CountryModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)  # type: ignore[union-attr]
        self.dropout = nn.Dropout(dropout)
        self.country_head = nn.Linear(hidden, len(COUNTRY_CLASSES))
        self.state_head = nn.Linear(hidden, len(STATE_CLASSES))

    @classmethod
    def from_base(cls, name: str) -> CountryModel:
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(name))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = self.dropout((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        return self.country_head(pooled), self.state_head(pooled)


def target_matrix(dists: Sequence[Mapping[str, float]], classes: Sequence[str]) -> torch.Tensor:
    index = {c: i for i, c in enumerate(classes)}
    m = torch.zeros(len(dists), len(classes))
    for r, d in enumerate(dists):
        for k, v in d.items():
            m[r, index[k]] = v
    return m


def soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()

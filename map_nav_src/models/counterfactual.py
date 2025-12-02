import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LatentCounterfactualGenerator(nn.Module):
    """Generate counterfactual latent features by perturbing salient semantic cues.

    The generator estimates token importance conditioned on the language context and
    injects semantic interventions on the most critical visual tokens while preserving
    background structure. This encourages the policy to disentangle causal cues from
    spurious correlations during training.
    """

    def __init__(
        self,
        hidden_size: int,
        top_ratio: float = 0.3,
        noise_scale: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.top_ratio = top_ratio
        self.noise_scale = noise_scale
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        return (tensor * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _estimate_importance(
        self, visual_embeds: torch.Tensor, visual_masks: torch.Tensor, text_embeds: torch.Tensor, text_masks: torch.Tensor
    ) -> torch.Tensor:
        text_context = self._masked_mean(text_embeds, text_masks)
        query = self.query_proj(text_context).unsqueeze(1)
        keys = self.key_proj(visual_embeds)
        scores = (keys * query).sum(-1) / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(visual_masks.logical_not(), -1e4)
        weights = F.softmax(scores, dim=-1)
        return weights

    def _sample_perturbation(self, weights: torch.Tensor, visual_masks: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = weights.size()
        perturb_mask = torch.zeros_like(weights, dtype=torch.bool)
        valid_lens = visual_masks.long().sum(dim=1)
        for i in range(batch_size):
            k = max(1, int(valid_lens[i].item() * self.top_ratio))
            topk = torch.topk(weights[i], k)
            perturb_mask[i].scatter_(0, topk.indices, True)
        perturb_mask = perturb_mask & visual_masks
        return perturb_mask

    def forward(
        self, visual_embeds: torch.Tensor, visual_masks: torch.Tensor, text_embeds: torch.Tensor, text_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        importance = self._estimate_importance(visual_embeds, visual_masks, text_embeds, text_masks)
        perturb_mask = self._sample_perturbation(importance, visual_masks)

        context_anchor = self._masked_mean(visual_embeds, visual_masks).unsqueeze(1)
        noise = torch.randn_like(visual_embeds) * self.noise_scale

        counterfactual = visual_embeds.clone()
        replace = context_anchor + noise
        counterfactual = torch.where(perturb_mask.unsqueeze(-1), self.dropout(replace), counterfactual)

        meta = {
            "weights": importance.detach(),
            "perturb_mask": perturb_mask,
            "factual": visual_embeds.detach(),
        }
        return counterfactual, meta

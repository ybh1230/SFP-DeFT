from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import DeFTConfig
from .density_tracing import DensityGuidedFeatureTracing


class DeFTSFPAdapter(nn.Module):
    """Small wrapper for inserting DeFT into an existing SFP-style segmentor.

    The preferred integration point is after SFP/CLIP has produced purified
    visual patch tokens and before image-text similarity is computed:

        tokens = sfp_visual_backbone(...)
        tokens = adapter.refine_visual_features(tokens, hw=(h, w))
        logits = text_image_similarity(tokens, text_features)

    This adapter intentionally avoids assumptions about the exact SFP repository
    API. If the wrapped model returns a dict containing ``visual_tokens`` or
    ``image_features``, the adapter refines that field in-place in the returned
    dict. For precise results, patch the backbone method directly using
    ``refine_visual_features``.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        config: Optional[DeFTConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.deft = DensityGuidedFeatureTracing(config)

    @torch.no_grad()
    def refine_visual_features(
        self,
        visual_features: torch.Tensor,
        hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.deft(visual_features, hw=hw)

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.base_model is None:
            raise RuntimeError(
                "DeFTSFPAdapter.forward requires a base_model. "
                "Use refine_visual_features for direct integration."
            )

        output = self.base_model(*args, **kwargs)
        hw = kwargs.get("hw", None)
        if isinstance(output, Dict):
            for key in ("visual_tokens", "image_features", "patch_tokens"):
                value = output.get(key)
                if isinstance(value, torch.Tensor) and value.dim() in (3, 4):
                    output[key] = self.refine_visual_features(value, hw=hw)
                    output["deft_enabled"] = True
                    return output
        return output

import os

import torch
from transformers import AutoModel, AutoProcessor


class SigLIPEncoder:
    def __init__(
        self, model_name: str | None = None, device: str | None = None
    ):
        self.model_name = model_name or os.getenv(
            "TEXT_ENCODER_MODEL", "google/siglip2-so400m-patch14-384"
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _coerce_features(self, features) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "text_embeds") and features.text_embeds is not None:
            return features.text_embeds
        return features[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Nhận nhiều text -> trả về các vector SigLIP2 đã L2-normalized."""
        if not texts:
            return []
        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = self._coerce_features(features)
            features = self._normalize_features(features)
        return features.cpu().tolist()

    def encode(self, text: str) -> list[float]:
        """Nhận text -> trả về vector SigLIP2 đã L2-normalized."""
        return self.encode_many([text])[0]

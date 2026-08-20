import torch
from transformers import AutoModel, AutoProcessor


class SigLIPEncoder:
    def __init__(
        self, model_name: str = "google/siglip2-so400m-patch14-384", device: str | None = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, text: str) -> list[float]:
        """Nhận text -> trả về vector"""
        inputs = self.processor(
            text=[text], return_tensors="pt", padding="max_length"
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                elif hasattr(features, "text_embeds") and features.text_embeds is not None:
                    features = features.text_embeds
                else:
                    features = features[0]
            # features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

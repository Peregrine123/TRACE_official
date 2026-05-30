"""
Embedding utilities:
- HashEmbedder: dependency-free for smoke tests
- HFEmbedder: transformer-based mean pooled embeddings
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


class HashEmbedder:
    """
    Hash-based embedding: not semantic, but stable for pipeline testing
    """

    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8", errors="ignore")).digest()

            vals = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            reps = int(np.ceil(self.dim / len(vals)))
            vec = np.tile(vals, reps)[: self.dim]
            vec = vec - vec.mean()
            norm = np.linalg.norm(vec) + 1e-8
            out[i] = vec / norm
        return out


class HFEmbedder:
    """
    transformer embeddings (mean pooling)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Ensure tokenizer has a pad token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            self.tok.pad_token_id = self.tok.eos_token_id
            self.tok.padding_side = "left"
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden, attn_mask):
        mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def embed(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            batch = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**batch)

            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = out[0]
            pooled = self._mean_pool(last_hidden, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            return pooled.detach().cpu().numpy().astype(np.float32)

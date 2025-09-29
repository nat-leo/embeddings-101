# openai_embed.py — Python 3.13

from __future__ import annotations

from typing_extensions import TypedDict, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ---------- Response typing (OpenAI-style) ----------

class EmbeddingObject(TypedDict):
    object: Literal["embedding"]
    embedding: list[float]
    index: int


class Usage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(TypedDict):
    object: Literal["list"]
    data: list[EmbeddingObject]
    model: str
    usage: Usage


# ---------- Adapter ----------

class OpenAiEmbed:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        dtype=np.float32,  # keep payloads small
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Optional tokenizer for usage.token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self.tokenizer = None
        self.normalize = normalize
        self.dtype = dtype

    def _to_batch(self, inp: str | list[str]) -> list[str]:
        return [inp] if isinstance(inp, str) else list(inp)

    def _count_tokens(self, texts: list[str]) -> int:
        if self.tokenizer is None:
            # Fallback: rough whitespace tokenization
            return sum(len(t.split()) for t in texts)
        enc = self.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        return sum(len(ids) for ids in enc["input_ids"])

    def create(self, input: str | list[str], model: str | None = None) -> EmbeddingsResponse:
        texts = self._to_batch(input)

        # Compute embeddings
        vecs = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(self.dtype, copy=False)

        data: list[EmbeddingObject] = []
        for i, v in enumerate(vecs):
            data.append({
                "object": "embedding",
                "embedding": v.tolist(),  # JSON-serializable list[float]
                "index": i,
            })

        prompt_tokens = self._count_tokens(texts)

        resp: EmbeddingsResponse = {
            "object": "list",
            "data": data,
            "model": model or self.model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
        return resp


if __name__ == "__main__":
    api = OpenAiEmbed()
    out = api.create(["hello world", "embeddings are vectors"])
    print(out["model"], len(out["data"]), "dims=", len(out["data"][0]["embedding"]))

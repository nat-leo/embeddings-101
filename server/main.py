import os
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Lazy import so container startup is snappy before first request
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Embeddings Server", version="1.0.0", docs_url="/")

_model = None
_dim = None

def get_model():
    global _model, _dim
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        # Warm-up with a tiny encode to determine dim and compile caches
        vec = _model.encode(["warmup"], normalize_embeddings=True)
        _dim = int(vec.shape[1])
    return _model

class EmbedRequest(BaseModel):
    inputs: List[str] = Field(..., description="Texts to embed")
    normalize: bool = True
    batch_size: int = Field(32, ge=1, le=1024)
    max_length: Optional[int] = Field(
        None, description="Optional hard truncate (tokens) if model supports it"
    )

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int
    normalized: bool
    device: str

@app.get("/healthz")
def healthz():
    m = get_model()
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "dim": _dim,
    }

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.inputs:
        raise HTTPException(status_code=400, detail="No inputs provided.")
    m = get_model()

    # sentence-transformers handles batching internally with batch_size
    kwargs = {}
    # Some models respect max_length via encode kwargs; harmless if ignored
    if req.max_length is not None:
        kwargs["max_length"] = req.max_length

    try:
        embs = m.encode(
            req.inputs,
            batch_size=req.batch_size,
            normalize_embeddings=req.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EmbedResponse(
        embeddings=np.asarray(embs, dtype=np.float32).tolist(),
        model=MODEL_NAME,
        dim=_dim or embs.shape[1],
        normalized=req.normalize,
        device=DEVICE,
    )

@app.get("/embed_one")
def embed_one(text: str, normalize: bool = True):
    resp = embed(EmbedRequest(inputs=[text], normalize=normalize))
    return resp

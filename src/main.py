from fastapi import FastAPI
from pydantic import BaseModel
import os

# Import the adapter AND its typed response (from the file where you defined it)
from adapters import OpenAiEmbed, EmbeddingsResponse  # TypedDicts from your adapter module

app = FastAPI(title="Embeddings Server", version="1.0.0", docs_url="/")

MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
model = OpenAiEmbed(MODEL_NAME)

# --- Request schema ---
class EmbeddingsRequest(BaseModel):
    model: str | None = None
    data: str | list[str]

# Tell FastAPI the response schema + return the precise type
@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def embed(req: EmbeddingsRequest) -> EmbeddingsResponse:
    return model.create(data=req.data, model=req.model)

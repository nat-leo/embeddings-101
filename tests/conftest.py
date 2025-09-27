import main as app_module
from fastapi.testclient import TestClient
import hashlib
import numpy as np
import pytest

DIM = 384

class FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def encode(
        self,
        inputs,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        **kwargs
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
        vecs = []
        for s in inputs:
            # Deterministic vector from SHA256 hash of text
            h = hashlib.sha256(s.encode("utf-8")).digest()
            # Repeat/trim to DIM
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            arr = np.tile(arr, int(np.ceil(DIM / arr.size)))[:DIM]
            # Center/scale a bit
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            if normalize_embeddings:
                n = np.linalg.norm(arr) + 1e-9
                arr = arr / n
            vecs.append(arr)
        out = np.stack(vecs, axis=0)
        return out if convert_to_numpy else out.tolist()

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # Patch global SentenceTransformer import used in app.get_model()
    monkeypatch.setattr(app_module, "SentenceTransformer", lambda *a, **k: FakeModel())
    # Reset cached model/dim between tests to avoid state leakage
    app_module._model = None
    app_module._dim = None
    yield

@pytest.fixture()
def client():
    import main
    return TestClient(main.app)

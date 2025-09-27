def test_embed_batch(client):
    payload = {
        "inputs": ["hello world", "hello world!", "fast small embeddings"],
        "normalize": True,
        "batch_size": 16
    }
    r = client.post("/embed", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["dim"] == 384
    assert data["normalized"] is True
    embs = data["embeddings"]
    assert len(embs) == len(payload["inputs"])
    assert len(embs[0]) == 384

def test_embed_one(client):
    r = client.get("/embed_one", params={"text": "quick brown fox", "normalize": True})
    assert r.status_code == 200
    data = r.json()
    assert data["dim"] == 384
    assert len(data["embeddings"]) == 1

def test_embed_empty_inputs(client):
    r = client.post("/embed", json={"inputs": []})
    assert r.status_code == 400
    assert "No inputs" in r.json()["detail"]

def test_normalization_toggle(client):
    # Call twice with and without normalization; norms should reflect flag
    text = "norm-check"
    r1 = client.post("/embed", json={"inputs": [text], "normalize": True})
    r2 = client.post("/embed", json={"inputs": [text], "normalize": False})
    v1 = r1.json()["embeddings"][0]
    v2 = r2.json()["embeddings"][0]

    import math
    n1 = math.sqrt(sum(x*x for x in v1))
    n2 = math.sqrt(sum(x*x for x in v2))

    assert abs(n1 - 1.0) < 1e-3   # normalized ≈ 1
    assert n2 > 1.0               # unnormalized larger than 1

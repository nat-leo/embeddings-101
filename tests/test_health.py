def test_health_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"]
    assert body["device"]
    assert body["dim"] == 384  # from FakeModel

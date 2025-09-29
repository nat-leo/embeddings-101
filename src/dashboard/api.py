import requests
import numpy as np

def embed(texts: list[str]) -> list[list[float]]:
    url = "http://0.0.0.0:80/v1/embeddings"
    payload = {"data": texts}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print(f"Response:\n{response}")
    else:
        print(f"Error {response.status_code}: {response.text}")

    return response

def to_2d(vectors: list[list[float]]) -> np.ndarray:
    X = np.asarray(vectors, dtype=float)
    n, d = X.shape
    if n == 1:
        # single point → put at origin
        return np.zeros((1, 2))
    Xc = X - X.mean(axis=0, keepdims=True)
    # If dimension or rank is too small, pad safely
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:2]
        Y = Xc @ comps.T
    except np.linalg.LinAlgError:
        Y = np.zeros((n, 2))
    # If only 1 component has variance, pad 2nd axis with zeros
    if Y.shape[1] == 1:
        Y = np.hstack([Y, np.zeros((n, 1))])
    if Y.shape[1] == 0:
        Y = np.zeros((n, 2))
    return Y[:, :2]

import requests

OLLAMA_EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"


def get_bge_m3_embedding(texts):
    """
    texts: List[str]
    return: List[List[float]]
    """
    payload = {
        "model": "bge-m3",
        "input": texts
    }

    resp = requests.post(OLLAMA_EMBEDDING_URL, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()["data"]
    return [item["embedding"] for item in data]

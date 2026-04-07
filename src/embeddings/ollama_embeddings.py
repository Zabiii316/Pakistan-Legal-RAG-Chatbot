import requests
from typing import List


class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434/api"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]
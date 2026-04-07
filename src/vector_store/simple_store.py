import math
from typing import List

from src.embeddings.ollama_embeddings import OllamaEmbeddings


class SimpleVectorStore:
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.embedding_client = OllamaEmbeddings(model="nomic-embed-text")

    def add_documents(self, docs: List[str]) -> None:
        self.documents.extend(docs)
        self.embeddings.extend(self.embedding_client.embed_batch(docs))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.documents:
            return []

        query_embedding = self.embedding_client.embed(query)

        scored = []
        for doc, embedding in zip(self.documents, self.embeddings):
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]
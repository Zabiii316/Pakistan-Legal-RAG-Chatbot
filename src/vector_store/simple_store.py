import re
from typing import List


class SimpleVectorStore:
    def __init__(self) -> None:
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]) -> None:
        self.documents.extend(docs)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\b[a-zA-Z0-9\-]+\b", text.lower())

        query_words = tokenize(query)

        scored = []
        for doc in self.documents:
            doc_text = doc.lower()
            doc_words = tokenize(doc)

            score = 0

            for word in query_words:
                if word in doc_words:
                    score += 3
                elif word in doc_text:
                    score += 1

            if query.lower() in doc_text:
                score += 5

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored if score > 0][:top_k]
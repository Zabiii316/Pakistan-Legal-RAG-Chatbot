from typing import List
import re


class SimpleVectorStore:
    def __init__(self):
        self.documents: List[str] = []

    def add_documents(self, documents: List[str]) -> None:
        self.documents.extend(documents)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_words = self._tokenize(query)
        scored = []

        for doc in self.documents:
            doc_words = self._tokenize(doc)
            doc_text = doc.lower()
            score = 0

            for word in query_words:
                if word in doc_words:
                    score += 3
                elif word in doc_text:
                    score += 1

            query_lower = query.lower()
            if "article" in query_lower and "article" in doc_text:
                score += 4
            if "section" in query_lower and "section" in doc_text:
                score += 4
            if "constitution" in query_lower and "constitution" in doc_text:
                score += 3
            if "penal code" in query_lower and "penal code" in doc_text:
                score += 3
            if query_lower in doc_text:
                score += 6

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        filtered = [doc for score, doc in scored if score > 0]
        return filtered[:top_k] if filtered else self.documents[:top_k]
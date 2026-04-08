from typing import Dict, List, Optional

from src.document_processor.loader import DocumentLoader
from src.document_processor.chunker import TextChunker
from src.vector_store.simple_store import SimpleVectorStore
from src.llm.ollama_client import OllamaClient


class RAGEngine:
    def __init__(self, raw_docs_dir: str):
        self.loader = DocumentLoader(raw_docs_dir)
        self.chunker = TextChunker(chunk_size=600, overlap=80)
        self.store = SimpleVectorStore()
        self.chunk_sources: List[str] = []
        self.llm = OllamaClient(model="gemma3:4b")
        self._indexed = False

    def build_index(self) -> None:
        documents = self.loader.load_txt_documents()

        chunks: List[str] = []
        sources: List[str] = []

        for doc in documents:
            doc_chunks = self.chunker.chunk_text(doc["text"])
            for chunk in doc_chunks:
                clean_chunk = chunk.strip()
                if clean_chunk:
                    chunks.append(clean_chunk)
                    sources.append(doc["filename"])

        if not chunks:
            chunks = [
                "Section 420 of Pakistan Penal Code relates to cheating and dishonestly inducing delivery of property.",
                "Article 25 of the Constitution ensures equality of citizens and equal protection of law.",
                "Bail can be granted under certain conditions in criminal cases depending on the facts and applicable law.",
                "Contract law in Pakistan is governed by the Contract Act 1872.",
            ]
            sources = ["builtin"] * len(chunks)

        self.store.add_documents(chunks)
        self.chunk_sources = sources
        self._indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not self._indexed:
            self.build_index()

        results = self.store.search(query, top_k=top_k)

        enriched: List[Dict[str, str]] = []
        for result in results:
            try:
                idx = self.store.documents.index(result)
                enriched.append({
                    "text": result,
                    "source": self.chunk_sources[idx]
                })
            except ValueError:
                enriched.append({
                    "text": result,
                    "source": "unknown"
                })

        return enriched

    def get_context(self, input_text: str, top_k: int = 3) -> Dict:
        context_items = self.retrieve(input_text, top_k=top_k)

        return {
            "relevant_context": [
                {
                    "source": item["source"],
                    "text": self._shorten(item["text"], 220)
                }
                for item in context_items
            ],
            "metadata": {
                "source_count": len(context_items),
                "sources": [item["source"] for item in context_items]
            }
        }

    def generate(self, input_text: str, history: Optional[List[Dict[str, str]]] = None) -> Dict:
        context_items = self.retrieve(input_text, top_k=3)
        answer = self.llm.chat(input_text, context_items, history=history or [])

        return {
            "answer": answer,
            "relevant_context": [
                {
                    "source": item["source"],
                    "text": self._shorten(item["text"], 220)
                }
                for item in context_items
            ],
            "metadata": {
                "source_count": len(context_items),
                "sources": [item["source"] for item in context_items]
            }
        }

    def stream_generate(self, input_text: str, history: Optional[List[Dict[str, str]]] = None):
        context_items = self.retrieve(input_text, top_k=3)
        return self.llm.stream_chat(input_text, context_items, history=history or [])

    def _shorten(self, text: str, limit: int = 220) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit] + "..."
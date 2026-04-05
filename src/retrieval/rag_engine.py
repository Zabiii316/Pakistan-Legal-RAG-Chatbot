from src.document_processor.loader import DocumentLoader
from src.document_processor.chunker import TextChunker
from src.vector_store.simple_store import SimpleVectorStore
from src.llm.ollama_client import OllamaClient


class RAGEngine:
    def __init__(self, raw_docs_dir: str):
        self.loader = DocumentLoader(raw_docs_dir)
        self.chunker = TextChunker(chunk_size=600, overlap=80)
        self.store = SimpleVectorStore()
        self.chunk_sources = []
        self.llm = OllamaClient(model="gemma3:4b")
        self._indexed = False

    def build_index(self) -> None:
        documents = self.loader.load_txt_documents()

        chunks = []
        sources = []

        for doc in documents:
            doc_chunks = self.chunker.chunk_text(doc["text"])
            for chunk in doc_chunks:
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
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

    def retrieve(self, query: str, top_k: int = 2):
        if not self._indexed:
            self.build_index()

        results = self.store.search(query, top_k=top_k)

        enriched = []
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

    def generate(self, input_text: str, history=None) -> dict:
        context_items = self.retrieve(input_text, top_k=2)
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
            },
        }

    def _shorten(self, text: str, limit: int = 220) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit] + "..."
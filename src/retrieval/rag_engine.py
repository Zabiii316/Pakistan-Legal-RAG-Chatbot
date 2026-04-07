from src.document_processor.loader import DocumentLoader
from src.document_processor.chunker import TextChunker
from src.vector_store.simple_store import SimpleVectorStore
from src.llm.ollama_client import OllamaClient
from src.llm.openai_fallback_client import OpenAIFallbackClient


class RAGEngine:
    def __init__(self, raw_docs_dir: str):
        self.loader = DocumentLoader(raw_docs_dir)
        self.chunker = TextChunker(chunk_size=600, overlap=80)
        self.store = SimpleVectorStore()
        self.chunk_sources = []
        self.llm = OllamaClient(model="gemma3:4b")
        self.openai_fallback = OpenAIFallbackClient(model="gpt-5.4-mini")
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

    def retrieve(self, query: str, top_k: int = 3):
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

    def _shorten(self, text: str, limit: int = 220) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit] + "..."

    def _build_prompt(self, query: str, context_items: list[dict], history=None) -> str:
        history = history or []

        history_text = ""
        if history:
            lines = []
            for turn in history[-4:]:
                lines.append(f"User: {turn['user']}")
                lines.append(f"Assistant: {turn['bot']}")
            history_text = "\n".join(lines)

        context_text = "\n\n".join(
            [
                f"Source: {item['source']}\nText: {item['text']}"
                for item in context_items
            ]
        )

        prompt = f"""
You are a careful legal assistant focused on Pakistan law.

Rules:
- Answer only from the provided legal context.
- Be accurate, direct, and concise.
- Do not copy large raw chunks unless necessary.
- Do not invent laws, punishments, article numbers, or case citations.
- If the context is insufficient, say exactly:
  "The provided context does not contain enough information to answer this reliably."

Conversation history:
{history_text if history_text else "No prior conversation."}

Retrieved legal context:
{context_text if context_text else "No legal context found."}

User question:
{query}

Write the answer in this format:

Answer:
<2 to 4 short sentences>

Legal Reference:
<relevant article or section if available>

Explanation:
<brief plain-language explanation>

Caution:
<one short caution sentence>
""".strip()

        return prompt

    def _is_local_context_strong(self, context_items: list[dict], query: str) -> bool:
        if not context_items:
            return False

        query_words = [w.lower() for w in query.split() if len(w) > 2]
        if not query_words:
            return True

        top_text = " ".join(item["text"] for item in context_items[:2]).lower()
        matches = sum(1 for word in query_words if word in top_text)

        return matches >= max(1, min(3, len(query_words) // 2))

    def generate(self, input_text: str, history=None) -> dict:
        context_items = self.retrieve(input_text, top_k=3)
        use_local = self._is_local_context_strong(context_items, input_text)

        if use_local:
            prompt = self._build_prompt(input_text, context_items, history=history)
            answer = self.llm.generate(prompt)
            source_mode = "local_rag"
        else:
            answer = self.openai_fallback.answer(input_text, history=history)
            source_mode = "openai_fallback"

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
                "sources": [item["source"] for item in context_items],
                "answer_mode": source_mode
            },
        }

    def stream_generate(self, input_text: str, history=None):
        context_items = self.retrieve(input_text, top_k=3)
        prompt = self._build_prompt(input_text, context_items, history=history)
        return self.llm.stream_generate(prompt)

    def get_relevant_context(self, input_text: str) -> list[dict]:
        context_items = self.retrieve(input_text, top_k=3)
        return [
            {
                "source": item["source"],
                "text": self._shorten(item["text"], 220)
            }
            for item in context_items
        ]
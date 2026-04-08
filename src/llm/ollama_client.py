import json
from typing import Dict, Generator, List, Optional

import requests


class OllamaClient:
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434/api"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _build_messages(
        self,
        query: str,
        context_items: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        context_block = "\n\n".join(
            [
                f"Source: {item['source']}\nText: {item['text']}"
                for item in context_items
            ]
        )

        history_text = ""
        if history:
            lines = []
            for turn in history[-4:]:
                lines.append(f"User: {turn['user']}")
                lines.append(f"Assistant: {turn['bot']}")
            history_text = "\n".join(lines)

        system_message = (
            "You are a careful legal assistant focused on Pakistan law.\n\n"
            "Rules:\n"
            "- Answer only from the provided legal context.\n"
            "- Be accurate, direct, and concise.\n"
            "- Do not dump long legal chunks.\n"
            "- Do not invent laws, punishments, case citations, or article numbers.\n"
            "- If the provided context is insufficient, say exactly: "
            "'The provided context does not contain enough information to answer this reliably.'\n"
            "- Use prior conversation only to understand follow-up references.\n"
            "- Keep the response clean and useful.\n\n"
            "Preferred format:\n"
            "Answer:\n"
            "<2 to 4 short sentences>\n\n"
            "Legal Reference:\n"
            "<relevant article or section if available>\n\n"
            "Explanation:\n"
            "<brief plain-language explanation>\n\n"
            "Caution:\n"
            "<one short caution sentence>"
        )

        user_message = f"""
Conversation history:
{history_text if history_text else "No prior conversation."}

User question:
{query}

Retrieved legal context:
{context_block if context_block else "No legal context found."}
""".strip()

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def chat(
        self,
        query: str,
        context_items: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model": self.model,
                "messages": self._build_messages(query, context_items, history),
                "stream": False,
                "options": {"temperature": 0.1},
                "keep_alive": "10m",
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    def stream_chat(
        self,
        query: str,
        context_items: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model": self.model,
                "messages": self._build_messages(query, context_items, history),
                "stream": True,
                "options": {"temperature": 0.1},
                "keep_alive": "10m",
            },
            timeout=180,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8").strip()
            if not decoded:
                continue

            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                continue

            token = data.get("message", {}).get("content", "")
            if token:
                yield token

            if data.get("done", False):
                break
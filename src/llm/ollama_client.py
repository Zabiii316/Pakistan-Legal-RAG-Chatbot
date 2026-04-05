import requests
from typing import List, Dict


class OllamaClient:
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434/api"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, query: str, context_items: List[Dict[str, str]], history: List[Dict[str, str]] | None = None) -> str:
        context_block = "\n\n".join(
            [
                f"Source: {item['source']}\nText: {item['text']}"
                for item in context_items
            ]
        )

        history_text = ""
        if history:
            history_lines = []
            for turn in history[-4:]:
                history_lines.append(f"User: {turn['user']}")
                history_lines.append(f"Assistant: {turn['bot']}")
            history_text = "\n".join(history_lines)

        system_message = (
            "You are a careful legal assistant focused on Pakistan law. "
            "Answer accurately, briefly, and only from the provided legal context. "
            "Use prior conversation only to understand follow-up references like "
            "'that article', 'previous section', or 'what about that'. "
            "Do not invent legal sections, articles, or case citations. "
            "If the context is insufficient, clearly say so. "
            "Keep the answer natural, concise, and helpful. "
            "Use this format:\n"
            "Answer:\n"
            "Legal Reference:\n"
            "Explanation:\n"
            "Caution:"
        )

        user_message = f"""
Conversation history:
{history_text if history_text else "No prior conversation."}

User question:
{query}

Retrieved legal context:
{context_block}
""".strip()

        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1
                },
                "keep_alive": "10m"
            },
            timeout=180
        )

        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
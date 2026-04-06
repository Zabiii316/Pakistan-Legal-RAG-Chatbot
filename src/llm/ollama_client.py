import requests
from typing import Dict, List


class OllamaClient:
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434/api"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        query: str,
        context_items: List[Dict[str, str]],
        history: List[Dict[str, str]] | None = None,
    ) -> str:
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
            "- Use prior conversation only to understand follow-up references like "
            "'that article', 'previous section', or 'what about that'.\n\n"
            "Response format:\n"
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
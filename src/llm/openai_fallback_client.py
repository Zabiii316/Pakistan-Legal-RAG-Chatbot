import os
from typing import Dict, List, Optional

from openai import OpenAI


class OpenAIFallbackClient:
    def __init__(self, model: str = "gpt-5.4-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None

    def is_enabled(self) -> bool:
        return self.client is not None

    def answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if not self.client:
            return (
                "Answer:\n"
                "The provided context does not contain enough information to answer this reliably, "
                "and OpenAI fallback is not configured.\n\n"
                "Legal Reference:\n"
                "No verified local source available.\n\n"
                "Explanation:\n"
                "Set OPENAI_API_KEY to enable broader general-information fallback.\n\n"
                "Caution:\n"
                "This is not legal advice."
            )

        history = history or []
        history_text = ""
        if history:
            lines = []
            for turn in history[-4:]:
                lines.append(f"User: {turn['user']}")
                lines.append(f"Assistant: {turn['bot']}")
            history_text = "\n".join(lines)

        instructions = (
            "You are a careful legal assistant focused on Pakistan law.\n\n"
            "Rules:\n"
            "- This is a fallback answer because the local uploaded legal file did not contain enough context.\n"
            "- Give general legal information only.\n"
            "- Be accurate, direct, and concise.\n"
            "- Do not pretend the answer came from the uploaded file.\n"
            "- If you are unsure, say so clearly.\n"
            "- Do not invent case citations.\n"
            "- Keep the response useful and cautious.\n\n"
            "Use this format:\n"
            "Answer:\n"
            "<2 to 5 short sentences>\n\n"
            "Legal Reference:\n"
            "<general legal reference if known, otherwise say not verified from local file>\n\n"
            "Explanation:\n"
            "<brief plain-language explanation>\n\n"
            "Caution:\n"
            "<one short caution sentence>"
        )

        user_input = f"""
Conversation history:
{history_text if history_text else "No prior conversation."}

User question:
{query}

Important:
The local uploaded legal dataset did not contain enough reliable context for this answer.
Respond as a general AI fallback, not as a document-grounded answer.
""".strip()

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
            )
            return getattr(response, "output_text", "").strip()

        except Exception as e:
            return (
                "Answer:\n"
                "The provided context does not contain enough information to answer this reliably, "
                "and the OpenAI fallback service is currently unavailable.\n\n"
                "Legal Reference:\n"
                "No verified local source available.\n\n"
                "Explanation:\n"
                f"Fallback service error: {str(e)}\n\n"
                "Caution:\n"
                "This is not legal advice."
            )
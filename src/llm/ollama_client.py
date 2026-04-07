import json
from typing import Iterator

import requests


class OllamaClient:
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434/api"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                },
                "keep_alive": "10m",
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def stream_generate(self, prompt: str) -> Iterator[str]:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                },
                "keep_alive": "10m",
            },
            timeout=180,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            token = data.get("response", "")
            if token:
                yield token

            if data.get("done", False):
                break
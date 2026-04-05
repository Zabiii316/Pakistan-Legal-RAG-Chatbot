from collections import defaultdict
from typing import Dict, List


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.sessions: Dict[str, List[dict]] = defaultdict(list)
        self.max_turns = max_turns

    def add_message(self, session_id: str, user_message: str, bot_response: str) -> None:
        self.sessions[session_id].append(
            {"user": user_message, "bot": bot_response}
        )

        if len(self.sessions[session_id]) > self.max_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns:]

    def get_history(self, session_id: str) -> List[dict]:
        return self.sessions.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        self.sessions[session_id] = []
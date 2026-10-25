from collections import OrderedDict, deque


class ConversationMemory:

    def __init__(self, window_size=5):

        self.chat_history = deque(
            maxlen=window_size
        )

    def save_context(
        self,
        question,
        answer
    ):

        self.chat_history.append({
            "question": question,
            "answer": answer
        })

    def get_history(self):

        history = ""

        for chat in self.chat_history:

            history += f"""
User: {chat['question']}
Assistant: {chat['answer']}
"""

        return history

class SessionMemoryStore:

    # One ConversationMemory per session_id. OrderedDict keeps the
    # least recently used session first, so we can evict it when full.

    def __init__(self, window_size=5, max_sessions=1000):

        self.window_size = window_size
        self.max_sessions = max_sessions

        self.sessions = OrderedDict()

    def get(self, session_id):

        if session_id in self.sessions:

            self.sessions.move_to_end(session_id)

            return self.sessions[session_id]

        memory = ConversationMemory(
            window_size=self.window_size
        )

        self.sessions[session_id] = memory

        if len(self.sessions) > self.max_sessions:

            self.sessions.popitem(last=False)

        return memory

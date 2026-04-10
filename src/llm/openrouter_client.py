import os
from typing import Callable

from llm.models import ModelInfo
import httpx, json

class OpenRouterClient:
    CHAT_COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.headers = headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def fetch(self, messages: list[dict], model_info: ModelInfo, callback_func: Callable[[str], None]):
        payload = {
            "model": model_info.slug,
            "messages": messages,
            "stream": True
        }

        buffer = ""
        with httpx.Client(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            with client.stream("POST", OpenRouterClient.CHAT_COMPLETION_URL, headers=self.headers, json=payload) as r:
                r.raise_for_status()
                for chunk in r.iter_text():
                    buffer += chunk
                    while True:
                        line_end = buffer.find('\n')
                        if line_end == -1:
                            break
                        line = buffer[:line_end].strip()
                        buffer = buffer[line_end + 1:]
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                data_obj = json.loads(data)
                                content = data_obj["choices"][0]["delta"].get("content")
                                if content:
                                    # print(content, end="", flush=True)
                                    callback_func(content)
                            except json.JSONDecodeError:
                                pass
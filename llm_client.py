import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
_MAX_TOKENS = 3000


class LLMClient:
    def __init__(self):
        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_API_KEY", _API_KEY),
                ("AZURE_OPENAI_ENDPOINT", _ENDPOINT),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Set them in your environment or a local .env file (see .env.example)."
            )
        self._client = OpenAI(api_key=_API_KEY, base_url=_ENDPOINT)
        self._model = _DEPLOYMENT

    def complete(self, system: str, user: str, temperature: float) -> str:
        completion = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=self._model,
            temperature=temperature,
            max_tokens=_MAX_TOKENS,
        )
        return completion.choices[0].message.content

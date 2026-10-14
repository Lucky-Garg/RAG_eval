"""
Custom LLM-as-a-judge for DeepEval using any OpenAI-compatible endpoint
(Gemini, Groq, Ollama...). Configured through .env, so no paid OpenAI key is needed.
"""
import json
import os
import time

from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of the model output (handles ```json fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in judge output: {text[:200]}")
    return json.loads(text[start:end + 1])


class JudgeLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL")
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

    def load_model(self):
        return self.client

    def _call(self, prompt: str) -> str:
        # Retry with backoff: free tiers return 429 when rate limits are hit
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                time.sleep(float(os.getenv("CALL_DELAY", "2")))   # gentle pacing
                return resp.choices[0].message.content
            except RateLimitError:
                wait = 15 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s ...")
                time.sleep(wait)
        raise RuntimeError("Rate limit: too many retries. Reduce test cases or wait a minute.")

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._call(prompt)
        text = self._call(prompt + "\n\nReturn ONLY a valid JSON object. No markdown, no extra text.")
        return schema(**_extract_json(text))

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name
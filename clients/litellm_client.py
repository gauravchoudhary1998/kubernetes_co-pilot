from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests import Response


@dataclass(frozen=True, slots=True)
class LiteLLMClient:
    """HTTP client for generating text via a LiteLLM proxy (OpenAI-compatible API)."""

    base_url: str
    model: str
    token: str = ""
    timeout_seconds: float = 120.0

    def generate(self, prompt: str) -> str:
        """Generate a response for the supplied prompt."""
        endpoint = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return self._extract_response_text(response)
        except requests.ConnectionError:
            return (
                "Unable to connect to LiteLLM at "
                f"{self.base_url}. Ensure the LiteLLM proxy is running."
            )
        except requests.Timeout:
            return "The request to LiteLLM timed out before the model returned a response."
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            return f"LiteLLM returned an HTTP error: {status_code}."
        except requests.RequestException as exc:
            return f"LiteLLM request failed: {exc}."

    def _headers(self) -> dict[str, str]:
        """Build request headers for LiteLLM calls."""
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _extract_response_text(self, response: Response) -> str:
        """Extract generated text from a LiteLLM API response."""
        try:
            data: Any = response.json()
        except ValueError:
            return "LiteLLM returned a malformed response: response body was not valid JSON."

        if not isinstance(data, dict):
            return "LiteLLM returned a malformed response: expected a JSON object."

        try:
            generated_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return "LiteLLM returned a malformed response: missing choices[0].message.content."

        if not isinstance(generated_text, str):
            return "LiteLLM returned a malformed response: content is not a string."

        return generated_text.strip()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests import Response


@dataclass(frozen=True, slots=True)
class OllamaClient:
    """HTTP client for generating text with a local Ollama model."""

    base_url: str
    model: str
    timeout_seconds: float = 120.0

    def generate(self, prompt: str) -> str:
        """Generate a response for the supplied prompt."""
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            return self._extract_response_text(response)
        except requests.ConnectionError:
            return (
                "Unable to connect to Ollama at "
                f"{self.base_url}. Ensure Ollama is running locally."
            )
        except requests.Timeout:
            return "The request to Ollama timed out before the model returned a response."
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            return f"Ollama returned an HTTP error: {status_code}."
        except requests.RequestException as exc:
            return f"Ollama request failed: {exc}."

    def _extract_response_text(self, response: Response) -> str:
        """Extract generated text from an Ollama API response."""
        try:
            data: Any = response.json()
        except ValueError:
            return "Ollama returned a malformed response: response body was not valid JSON."

        if not isinstance(data, dict):
            return "Ollama returned a malformed response: expected a JSON object."

        generated_text = data.get("response")
        if not isinstance(generated_text, str):
            return "Ollama returned a malformed response: missing string field 'response'."

        return generated_text.strip()

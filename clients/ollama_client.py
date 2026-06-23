from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests
from requests import Response


@dataclass(frozen=True, slots=True)
class OllamaClient:
    """HTTP client for generating text with a local Ollama model."""

    base_url: str
    model: str
    token: str = ""
    timeout_seconds: float = 120.0

    def generate(self, prompt: str) -> str:
        """Generate a response for the supplied prompt."""
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        headers = self._headers()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
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

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield tokens as they stream from Ollama."""
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": True}

        try:
            with requests.post(
                endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data: Any = json.loads(line)
                    except ValueError:
                        continue
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except requests.ConnectionError:
            yield f"Unable to connect to Ollama at {self.base_url}. Ensure Ollama is running locally."
        except requests.Timeout:
            yield "The request to Ollama timed out before the model returned a response."
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            yield f"Ollama returned an HTTP error: {status_code}."
        except requests.RequestException as exc:
            yield f"Ollama request failed: {exc}."

    def _headers(self) -> dict[str, str]:
        """Build request headers for Ollama calls."""
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

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

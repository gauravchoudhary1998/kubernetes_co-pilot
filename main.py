import argparse
import logging

from clients.ollama_client import OllamaClient
from services.kubernetes_investigator import (
    KubernetesInvestigationError,
    KubernetesInvestigator,
)
from services.troubleshooting_copilot import TroubleshootingCopilot


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


def main() -> None:
    """Run an interactive Kubernetes pod troubleshooting session."""
    args = _parse_args()
    _configure_logging(verbose=args.verbose)

    ollama_client = OllamaClient(
        base_url=args.base_url,
        model=args.model,
    )
    copilot = TroubleshootingCopilot(ollama_client=ollama_client)
    investigator = KubernetesInvestigator()
    namespace, pod_name = _prompt_for_pod()

    print(f"\nCollecting Kubernetes evidence for pod {namespace}/{pod_name}...\n")
    try:
        context = investigator.investigate_pod(namespace=namespace, pod_name=pod_name)
    except KubernetesInvestigationError as exc:
        raise SystemExit(str(exc)) from exc

    print("Analyzing the collected evidence...\n")
    analysis = copilot.analyze(context)
    print(analysis)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the troubleshooting copilot."""
    parser = argparse.ArgumentParser(
        description="Analyze Kubernetes troubleshooting context with local Ollama."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _prompt_for_pod() -> tuple[str, str]:
    """Prompt the user for the Kubernetes pod to investigate."""
    namespace = _ask_required("Which namespace is the failing pod in?")
    pod_name = _ask_required("Which pod is failing?")
    return namespace, pod_name


def _ask_required(prompt: str) -> str:
    """Prompt until a non-empty answer is provided."""
    while True:
        answer = input(f"{prompt}\n> ").strip()
        if answer:
            return answer
        print("Please provide a value.")


def _configure_logging(verbose: bool) -> None:
    """Configure process-wide logging for the CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    main()

import argparse
import logging

from clients.ollama_client import OllamaClient
from models.remediation_plan import RemediationPlan
from services.kubernetes_investigator import (
    KubernetesInvestigationError,
    KubernetesInvestigator,
)
from services.remediation_executor import (
    RemediationExecutionError,
    RemediationExecutor,
)
from services.remediation_candidate_generator import RemediationCandidateGenerator
from services.remediation_planner import RemediationPlanner
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
    candidate_generator = RemediationCandidateGenerator()
    remediation_planner = RemediationPlanner()
    namespace, pod_name = _prompt_for_pod()

    print(f"\nCollecting Kubernetes evidence for pod {namespace}/{pod_name}...\n")
    try:
        context = investigator.investigate_pod(namespace=namespace, pod_name=pod_name)
    except KubernetesInvestigationError as exc:
        raise SystemExit(str(exc)) from exc

    candidate_set = candidate_generator.generate(context)
    print(f"Detected failure class: {candidate_set.classification}\n")
    print("Analyzing the collected evidence...\n")
    analysis = copilot.analyze(
        context=context,
        remediation_candidates=candidate_set.candidates,
    )
    print(analysis)

    remediation_plan = remediation_planner.parse(
        llm_response=analysis,
        context=context,
        candidates=candidate_set.candidates,
    )
    _handle_remediation_plan(remediation_plan)


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


def _handle_remediation_plan(remediation_plan: RemediationPlan) -> None:
    """Prompt for approval and execute eligible remediation actions."""
    if remediation_plan.parse_error:
        print(f"\nRemediation automation unavailable: {remediation_plan.parse_error}")
        return

    if not remediation_plan.actions:
        print("\nNo remediation actions were proposed for automation.")
        return

    executor: RemediationExecutor | None = None

    for action in remediation_plan.actions:
        print("\nProposed remediation action:")
        print(f"  Candidate: {action.candidate_id}")
        print(f"  Action: {action.action_type}")
        print(f"  Category: {action.action_category}")
        print(f"  Target: {action.target_kind} {action.namespace}/{action.target_name}")
        print(f"  Risk: {action.risk_level}")
        print(f"  Solves root cause: {action.solves_root_cause}")
        print(f"  Description: {action.description or '<none>'}")
        print(f"  Rationale: {action.rationale or '<none>'}")

        if action.risk_level == "HIGH":
            print("  Automation: skipped because risk is HIGH.")
            continue

        if not action.executable:
            print(f"  Automation: unavailable. Reason: {action.policy_reason}")
            continue

        if not _confirm_action("Do you want me to apply this action?"):
            print("  Automation: skipped by user.")
            continue

        if executor is None:
            try:
                executor = RemediationExecutor()
            except RemediationExecutionError as exc:
                print(f"  Automation unavailable: {exc}")
                continue

        result = executor.execute(action)
        print(f"  Result: {result.message}")


def _confirm_action(prompt: str) -> bool:
    """Return True when the user explicitly approves an action."""
    answer = input(f"{prompt} [y/N]\n> ").strip().lower()
    return answer in {"y", "yes"}


def _configure_logging(verbose: bool) -> None:
    """Configure process-wide logging for the CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    main()

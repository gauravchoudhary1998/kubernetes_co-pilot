# Kubernetes Troubleshooting Copilot

Local, CLI-based Kubernetes pod troubleshooting assistant powered by Ollama and
the Kubernetes Python SDK.

The app asks which pod is failing, automatically collects deterministic evidence
from the active Kubernetes context, then asks a local LLM to analyze only that
evidence.

## Requirements

- Python 3.12+
- Ollama running locally at `http://localhost:11434`
- Ollama model `qwen3:8b`
- Local kubeconfig with access to the target namespace and pod

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install and run the local model:

```bash
ollama pull qwen3:8b
ollama serve
```

## Usage

Start the interactive troubleshooting session:

```bash
python main.py
```

The prompts are:

```text
Which namespace is the failing pod in?
>
Which pod is failing?
>
```

The investigator then gathers pod details, pod conditions, resource requests and
limits, probes, volumes, tolerations, per-container status reasons, events, logs,
and previous logs when the pod state calls for them.

Override Ollama settings if needed:

```bash
python main.py --base-url http://localhost:11434 --model qwen3:8b
```

## Project Structure

```text
.
├── clients/
│   └── ollama_client.py
├── models/
│   └── investigation_context.py
├── services/
│   ├── kubernetes_investigator.py
│   └── troubleshooting_copilot.py
├── main.py
├── requirements.txt
└── README.md
```

## Notes

This project intentionally uses a small, framework-free structure:

- `OllamaClient` handles HTTP communication with Ollama.
- `KubernetesInvestigator` collects deterministic pod evidence through the Kubernetes SDK.
- `InvestigationContext` stores collected pod evidence.
- `TroubleshootingCopilot` builds the LLM prompt and returns the analysis.

The LLM does not decide what cluster data to collect. Python code gathers the
evidence first, then the model analyzes that evidence.

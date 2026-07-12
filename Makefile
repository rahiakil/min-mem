.PHONY: benchmark figures baselines retrieval all test improve agent install

VENV := .venv/bin

benchmark:
	$(VENV)/python experiments/run_benchmark.py

baselines: benchmark
	$(VENV)/python experiments/benchmark_baselines.py
	$(VENV)/python experiments/generate_baseline_figure.py

retrieval: baselines
	$(VENV)/python experiments/retrieval_fidelity.py

figures: benchmark
	$(VENV)/python experiments/generate_figures.py
	$(VENV)/python experiments/sync_docs.py

improve:
	$(VENV)/python experiments/improve_loop.py

agent:
	$(VENV)/python -c "from agents.minimal_agent import MinimalAgent, AgentConfig; import json; from pathlib import Path; c=json.loads(Path('experiments/corpus.json').read_text())['samples']; m=[s['text'] for s in c]*4; a=MinimalAgent(m, AgentConfig()); print(json.dumps(a.compare_context(), indent=2))"

all: baselines retrieval figures

test:
	$(VENV)/pytest -v

install:
	python3 -m venv .venv
	$(VENV)/pip install -e ".[dev,experiments]"

package:
	$(VENV)/python -m build

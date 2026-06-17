.PHONY: benchmark figures paper all test improve agent

VENV := .venv/bin

benchmark:
	$(VENV)/python experiments/run_benchmark.py

figures: benchmark
	$(VENV)/python experiments/generate_figures.py

improve:
	$(VENV)/python experiments/improve_loop.py

agent:
	$(VENV)/python -c "from agents.minimal_agent import MinimalAgent, AgentConfig; import json; from pathlib import Path; c=json.loads(Path('experiments/corpus.json').read_text())['samples']; m=[s['text'] for s in c]*4; a=MinimalAgent(m, AgentConfig()); print(json.dumps(a.compare_context(), indent=2))"

paper: figures
	$(MAKE) -C ../min-mem-paper paper

all: improve paper

test:
	$(VENV)/pytest -v

install:
	python3 -m venv .venv
	$(VENV)/pip install -e ".[dev,experiments]"

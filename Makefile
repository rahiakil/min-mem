.PHONY: benchmark figures paper all test

VENV := .venv/bin

benchmark:
	$(VENV)/python experiments/run_benchmark.py

figures: benchmark
	$(VENV)/python experiments/generate_figures.py

agent-demo:
	$(VENV)/python experiments/agent_context_demo.py

paper: figures
	$(MAKE) -C ../min-mem-paper paper

all: figures paper

test:
	$(VENV)/pytest -v

install:
	python3 -m venv .venv
	$(VENV)/pip install -e ".[dev,experiments]"

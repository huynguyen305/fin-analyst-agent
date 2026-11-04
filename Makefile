.PHONY: install test demo
install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev,vn]"
test:
	.venv/bin/python -m pytest -q
demo:
	FIN_AGENT_FIXTURES=tests/fixtures .venv/bin/python -m fin_agent "Analyze DEMO, standard depth"

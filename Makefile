.PHONY: install install-kb test demo kb-index kb-search
install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev,vn]" && (test -d ../agent-hub && .venv/bin/pip install -e ../agent-hub --no-deps || true)
install-kb:
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -e ".[kb]"
kb-index:
	.venv/bin/python -m fin_agent.knowledge index
kb-search:
	.venv/bin/python -m fin_agent.knowledge search "$(Q)"
test:
	.venv/bin/python -m pytest -q
demo:
	FIN_AGENT_FIXTURES=tests/fixtures .venv/bin/python -m fin_agent "Analyze DEMO, standard depth"

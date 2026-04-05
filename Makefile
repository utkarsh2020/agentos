.PHONY: run test lint clean install help

PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3.12 2>/dev/null || command -v python3)

help:
	@echo ""
	@echo "  AgentOS — make targets"
	@echo ""
	@echo "  make run      Start the daemon (foreground)"
	@echo "  make test     Run the full test suite (40 tests)"
	@echo "  make lint     Run pyflakes static analysis"
	@echo "  make install  Run the system installer (requires sudo)"
	@echo "  make clean    Remove runtime artefacts"
	@echo ""

run:
	$(PYTHON) agentd/daemon.py

test:
	$(PYTHON) tests/test_suite.py

lint:
	@$(PYTHON) -m pyflakes agentd/ bin/agent 2>&1 || true

install:
	sudo bash deploy/install.sh

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f agentd.db agentd.db-wal agentd.db-shm agentd.pid agentd.sock
	rm -f logs/*.jsonl vault/*.enc vault/master.key

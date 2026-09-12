UV ?= uv
PYTHON_PATHS = scripts training packages/aeroguard-connectors/src

.PHONY: sync check lock-check format format-check lint typecheck test

sync:
	$(UV) sync --locked

lock-check:
	$(UV) lock --check

format:
	$(UV) run --locked ruff check --select I --fix $(PYTHON_PATHS)
	$(UV) run --locked ruff format $(PYTHON_PATHS)

format-check:
	$(UV) run --locked ruff format --check $(PYTHON_PATHS)

lint:
	$(UV) run --locked ruff check $(PYTHON_PATHS)

typecheck:
	$(UV) run --locked ty check --error-on-warning

test:
	$(UV) run --locked python -m pytest -q

check: lock-check format-check lint typecheck test

UV ?= uv
PYTHON_PATHS = scripts training packages/aeroguard-connectors/src

.PHONY: sync check lock-check format format-check lint typecheck test pipeline smoke-pipeline

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

pipeline:
	$(UV) run python -m scripts.run_pipeline --epochs 3 --batch-size 16 --lr 2e-4 --save-dir models/aeroguard_tslm

smoke-pipeline:
	$(UV) run python -m scripts.run_pipeline --max-steps 5 --save-dir models/test_smoke_pipeline

check: lock-check format-check lint typecheck test


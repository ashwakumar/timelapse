# Pull branch sid, then: make setup && make hf && make demo
PYTHON ?= .venv/bin/python
PORT ?= 8765

.PHONY: setup hf check demo eval

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -r requirements-training.txt
	@test -f .env || (cp .env.example .env && echo "Created .env — add your HF_TOKEN")

hf:
	@test -f .env || (echo "Copy .env.example to .env and set HF_TOKEN" && exit 1)
	set -a && . ./.env && set +a && $(PYTHON) scripts/prepare_hf_models.py

check:
	@test -f artifacts/run-opentslm-3ep/best.pt || (echo "Missing 3-pass best.pt. git pull origin sid" && exit 1)
	@test -f artifacts/run-opentslm-50ep/best.pt || (echo "Missing 50-pass best.pt. git pull origin sid" && exit 1)
	@test -f artifacts/run-opentslm-50ep-b/best.pt || (echo "Missing second 50-pass best.pt. git pull origin sid" && exit 1)
	@test -f data/surgical_telemetry/manifest.jsonl || (echo "Missing telemetry fixture. git pull origin sid" && exit 1)
	@ls -lh artifacts/run-opentslm-3ep/best.pt artifacts/run-opentslm-50ep/best.pt artifacts/run-opentslm-50ep-b/best.pt

demo: check
	@test -f .env || (echo "Copy .env.example to .env and set HF_TOKEN" && exit 1)
	set -a && . ./.env && set +a && $(PYTHON) scripts/run_demo.py --preload --port $(PORT)

eval: check
	@test -f .env || (echo "Copy .env.example to .env and set HF_TOKEN" && exit 1)
	set -a && . ./.env && set +a && $(PYTHON) scripts/run_demo.py --preload --port $(PORT)

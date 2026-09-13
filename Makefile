# Real training: see NEBIUS_TRAINING.md and vitaldb-* targets below.
# Legacy synthetic demo: make setup && make hf && make demo.
PYTHON ?= .venv/bin/python
PORT ?= 8765
VITALDB_PYTHON ?= .venv-vitaldb/bin/python
VITALDB_DATA ?= data/vitaldb/onset_v2
VITALDB_RUN ?= artifacts/run-vitaldb
VITALDB_BUNDLE ?= artifacts/vitaldb-bundles/vitaldb-run
VITALDB_EVAL ?= artifacts/vitaldb-evaluation
CASES ?= 1000
EPOCHS ?= 5
DEVICE ?= cuda

# Real VitalDB/OpenTSLM path. See NEBIUS_TRAINING.md; never creates fixtures.
.PHONY: vitaldb-setup vitaldb-data vitaldb-check vitaldb-train vitaldb-resume vitaldb-export vitaldb-eval
vitaldb-setup:
	python3.12 -m venv .venv-vitaldb
	$(VITALDB_PYTHON) -m pip install --upgrade pip
	$(VITALDB_PYTHON) -m pip install -r requirements-vitaldb-training.txt
	$(VITALDB_PYTHON) -m pip check

vitaldb-data:
	$(VITALDB_PYTHON) scripts/prepare_vitaldb_training.py --case-limit $(CASES) --output-dir $(VITALDB_DATA)

vitaldb-check:
	$(VITALDB_PYTHON) -m training.train_vitaldb --data-dir $(VITALDB_DATA) --preflight-only

vitaldb-train:
	$(VITALDB_PYTHON) -u -m training.train_vitaldb --data-dir $(VITALDB_DATA) --output-dir $(VITALDB_RUN) --epochs $(EPOCHS) --device $(DEVICE)

vitaldb-resume:
	$(VITALDB_PYTHON) -u -m training.train_vitaldb --data-dir $(VITALDB_DATA) --output-dir $(VITALDB_RUN) --epochs $(EPOCHS) --device $(DEVICE) --resume-from $(VITALDB_RUN)/last.pt

vitaldb-export:
	$(VITALDB_PYTHON) scripts/export_vitaldb_run.py --run-dir $(VITALDB_RUN) --data-dir $(VITALDB_DATA) --output $(VITALDB_BUNDLE) --archive

vitaldb-eval:
	$(VITALDB_PYTHON) -u -m evaluation.evaluate_vitaldb --bundle $(VITALDB_BUNDLE) --output-dir $(VITALDB_EVAL) --device $(DEVICE) --score-candidates

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

eval: vitaldb-eval

.PHONY: setup start stop test e2e integration dashboard monitoring session all_tests performance_test performance_dashboard model_comparison_dashboard model_download model_download_active model_download_all model_list model_promote model_compare orchestrate train registry_reset

PYTHON=.venv/bin/python

setup:
	./scripts/setup.sh

start:
	STARTUP_MODE=start ./scripts/setup.sh
	ENV=prod $(PYTHON) -m counter.entrypoints.webapp

stop:
	-docker stop test-mongo


# RUN ALL TESTS

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests -v

test_domain:
	PYTHONPATH=. $(PYTHON) -m pytest tests/domain -v

test_integration:
	PYTHONPATH=. $(PYTHON) -m pytest tests/integration -v

test_e2e:
	PYTHONPATH=. $(PYTHON) -m pytest tests/e2e -v

test_entrypoints:
	PYTHONPATH=. $(PYTHON) -m pytest tests/entrypoints -v

dashboard:
ifneq ($(filter monitoring,$(MAKECMDGOALS)),)
	PYTHONPATH=. $(PYTHON) monitoring/generate_monitoring_dashboard.py
	open tmp/dashboard/monitoring_dashboard.html
else ifneq ($(filter session,$(MAKECMDGOALS)),)
	PYTHONPATH=. $(PYTHON) monitoring/generate_session_dashboard.py
	open tmp/dashboard/session_dashboard.html
else
	@echo "Usage: make dashboard monitoring   (overall metrics)"
	@echo "       make dashboard session      (per-session debug view)"
endif

# no-op targets so `make dashboard monitoring` / `make dashboard session`
# are treated as arguments to the dashboard target above, not real goals.
monitoring session:
	@:


# PERFORMANCE TESTING
#
# Validates a model against model_management/evaluation/golden_dataset (Roboflow's
# own test split) via Ultralytics' validation pipeline, and pops up a dashboard.
# Also refreshes model_comparison_dashboard.html so a solo run's numbers show
# up there too. Override the model under test with:
#   make performance_test MODEL_PATH=tmp/models/other.pt
#   make performance_test MODEL_NAME=yolo26_v2   (name from model_management/model_registry/registry.json - ".pt" suffixes/pasted links are fine)

performance_test:
	$(if $(MODEL_PATH),MODEL_PATH=$(MODEL_PATH)) $(if $(MODEL_NAME),MODEL_NAME=$(MODEL_NAME)) $(PYTHON) model_management/evaluation/scripts/run_performance_tests.py

# Shows the latest run for one model (MODEL=name), or the most recently
# tested model overall when MODEL is omitted.
performance_dashboard:
	$(PYTHON) model_management/evaluation/dashboard/generate_performance_dashboard.py $(if $(MODEL),--model $(MODEL))
	open tmp/dashboard/performance_dashboard.html

model_comparison_dashboard:
	$(PYTHON) model_management/evaluation/dashboard/generate_performance_dashboard.py --comparison
	open tmp/dashboard/model_comparison_dashboard.html


# MODEL REGISTRY
#
# Models are tracked in model_management/model_registry/registry.json (name, source, active flag, use case).

# Downloads every registered model by default. MODELS="a b" downloads just
# those; `make model_download_active` downloads only the active model.
model_download:
	PYTHONPATH=. $(PYTHON) scripts/download_model.py $(if $(MODELS),--models $(MODELS))

model_download_active:
	PYTHONPATH=. $(PYTHON) scripts/download_model.py --active

model_download_all:
	PYTHONPATH=. $(PYTHON) scripts/download_model.py --all

model_list:
	PYTHONPATH=. $(PYTHON) -m model_management.model_registry.registry list

# Clears test_dataset version history and every model's cached performance in
# registry.json (keeps registered models) - for starting fresh with a new,
# unrelated golden dataset/use case. Also runs automatically during `make
# start` when CLEAR_MODEL_REGISTRY=true is set in .env.
registry_reset:
	PYTHONPATH=. $(PYTHON) -m model_management.model_registry.registry reset

# Set the active model directly, e.g.: make model_promote MODEL=yolo26_v2
model_promote:
	PYTHONPATH=. $(PYTHON) -m model_management.model_registry.registry set-active $(MODEL)

# Tests every registered model (or MODELS="yolo26 yolo26_v2") and reports
# each one's metrics side by side, ranked by METRIC (precision/recall/
# weighted_score, default weighted_score - see model_management/promotion.py).
# Add PROMOTE=1 to make the top-ranked model the active model automatically.
# Models last evaluated against a different/incompatible golden dataset are
# re-tested automatically before comparing.
model_compare:
	PYTHONPATH=. $(PYTHON) model_management/evaluation/dashboard/compare_models.py $(if $(MODELS),--models $(MODELS)) $(if $(METRIC),--metric $(METRIC)) $(if $(PROMOTE),--promote)

# Full train -> register -> evaluate -> promote pipeline (see
# model_management/orchestrator.py --help for every argument, including
# --use-case). Dataset/hyperparameter defaults come from model_management/config.py -
# ARGS is only needed for one-off overrides or --model-name. If the golden
# dataset changed since the last run, every registered model is re-compared,
# not just this one. Example:
#   make orchestrate ARGS="--model-name yolo26_v7"
orchestrate:
	PYTHONPATH=. $(PYTHON) model_management/orchestrator.py $(ARGS)

# Just download a dataset + train + register in registry.json - no
# evaluation/promotion (see model_management/training/yolo_training/train.py
# --help for every argument). Dataset/hyperparameter defaults come from
# model_management/config.py. Example:
#   make train ARGS="--model-name yolo26_experiment"
train:
	PYTHONPATH=. $(PYTHON) model_management/training/yolo_training/train.py $(ARGS)
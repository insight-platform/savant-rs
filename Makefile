export PROJECT_DIR=$(CURDIR)

# ── .envrc auto-loading ──────────────────────────────────────────────────────
# Source .envrc at parse time so every recipe inherits SKIA_BINARIES_URL &
# SCCACHE_CACHE_SIZE.  Wrapping in a function lets `return` work inside
# .envrc; stdout is discarded to keep its echo from polluting $(shell) output.
# Update the printf/export list below when .envrc adds new exports.
_ENVRC := $(shell bash -c 'f(){ source $(CURDIR)/.envrc; }; f >/dev/null && printf "%s %s" "$$SKIA_BINARIES_URL" "$$SCCACHE_CACHE_SIZE"')
ifneq ($(_ENVRC),)
  export SKIA_BINARIES_URL  := $(firstword $(_ENVRC))
  export SCCACHE_CACHE_SIZE := $(lastword $(_ENVRC))
  $(info .envrc loaded: SKIA_BINARIES_URL=$(SKIA_BINARIES_URL) SCCACHE_CACHE_SIZE=$(SCCACHE_CACHE_SIZE))
endif

# Project-local virtualenv.  All pip / python3 / pytest invocations go
# through VENV_BIN so we never accidentally touch the system site-packages.
VENV_DIR  ?= /opt/venv
VENV_BIN  := $(VENV_DIR)/bin
PYTHON    := $(VENV_BIN)/python3
PIP       := $(VENV_BIN)/pip
PYTEST    := $(VENV_BIN)/python3 -m pytest
# Passed to utils/build.sh so maturin/PyO3 use the same interpreter as pip/pytest.
# Override with `make PYTHON_INTERPRETER= dev` to let build.sh fall back (e.g. ./venv, maturin -f).
PYTHON_INTERPRETER ?= $(VENV_BIN)/python3

export PYTHON_VERSION=$(shell $(PYTHON) -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')

# Optional Cargo features for savant_python (e.g. SAVANT_FEATURES=gst).
# Multiple features can be comma-separated: SAVANT_FEATURES=gst,deepstream
SAVANT_FEATURES ?=

SP_DIR=$(PROJECT_DIR)/savant_python

.PHONY: docs build_savant build_savant_release clean tests bench reformat \
        sp-dev sp-install sp-pytest \
        all-dev all-release \
        fmt clippy lint \
        docker-build-docs serve-docs \
        deepstream-tests

dev: clean build_savant
release: clean build_savant_release

# -- formatting & linting ------------------------------------------------------

fmt:
	@echo "Running cargo fmt..."
	cargo fmt --all
	@echo "Running ruff format..."
	ruff format $(SP_DIR)/pytests 2>/dev/null || true

clippy:
	@echo "Running clippy on default members..."
	cargo clippy --all-targets -- -D warnings

lint: fmt clippy
	@echo "Running ruff check..."
	ruff check $(SP_DIR)/pytests --fix 2>/dev/null || true
	@echo "Lint complete."

# -- aggregate targets: build + test + install everything ---------------------

all-dev: fmt clippy lint dev install

all-release: fmt clippy lint release install

# -----------------------------------------------------------------------------

install-with-optional-deps:
	@WHL_NAME=$$(ls $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl); \
	echo "Installing $$WHL_NAME[clientsdk]"; \
	$(PIP) install --force-reinstall "$$WHL_NAME[clientsdk]"; \
	echo "Installed $$WHL_NAME[clientsdk]"

install:
	@WHL_NAME=$$(ls $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl); \
	echo "Installing $$WHL_NAME"; \
	$(PIP) install --force-reinstall "$$WHL_NAME"; \
	echo "Installed $$WHL_NAME"

# -- savant_python (savant_rs wheel) ------------------------------------------

sp-pytest: build_savant install
	@echo "Running savant_python Python tests..."
	cd $(SP_DIR) && $(PYTEST) pytests/ -v --tb=short

docker-build-docs:
	docker build -f docker/Dockerfile.docs -t savant-rs-docs .
	mkdir -p $(PROJECT_DIR)/docs/build/html
	docker run --rm --entrypoint cat savant-rs-docs /opt/docs-artifact.tar > $(PROJECT_DIR)/docs/build/html.tar
	tar --dereference --hard-dereference --directory $(PROJECT_DIR)/docs/build/html -xvf $(PROJECT_DIR)/docs/build/html.tar

serve-docs: docker-build-docs
	@echo "Serving docs at http://localhost:8080"
	docker run -it --rm -p 8080:80 -v $(PROJECT_DIR)/docs/build/html:/usr/share/nginx/html:ro nginx:alpine

docs: dev install
	@echo "Building docs..."
	cd $(PROJECT_DIR)/docs && LC_ALL=C.utf8 PATH="$(VENV_BIN):$$PATH" make clean html
	tar --dereference --hard-dereference --directory $(PROJECT_DIR)/docs/build/html -cvf $(PROJECT_DIR)/docs-artifact.tar .

build_savant:
	@echo "Building..."
	PYTHON_INTERPRETER=$(PYTHON_INTERPRETER) SAVANT_FEATURES=$(SAVANT_FEATURES) utils/build.sh debug

build_savant_release:
	@echo "Building..."
	PYTHON_INTERPRETER=$(PYTHON_INTERPRETER) SAVANT_FEATURES=$(SAVANT_FEATURES) utils/build.sh release

clean:
	@echo "Cleaning..."
	rm -rf $(PROJECT_DIR)/dist/*.whl

core-tests:
	@echo "Running core lib tests..."
	cd savant_core/savant_core && cargo build && cargo test -- --test-threads=1 # --show-output --nocapture

# savant_deepstream workspace members (Cargo package names). Each crate is tested in its own
# `cargo test -p …` so link steps stay small (avoids OOM on RAM-limited Jetson). Any failure
# aborts the rest and fails the make target. GPU-heavy runs may need, e.g.:
#   env -u DISPLAY EGL_PLATFORM=device make deepstream-tests
DEEPSTREAM_TEST_CRATES := \
	savant-deepstream-buffers \
	savant-deepstream-decoders \
	savant-deepstream-encoders \
	savant-deepstream-inputs \
	savant-nvidia-gpu-utils \
	savant-deepstream-nvinfer \
	savant-deepstream-nvtracker \
	savant-picasso

deepstream-tests:
	@echo "Running savant_deepstream crate tests (one package per cargo invocation)..."
	@set -e; cd $(PROJECT_DIR); \
	for p in $(DEEPSTREAM_TEST_CRATES); do \
		echo "==> cargo test -p $$p"; \
		cargo test -p "$$p" -- --test-threads=1; \
	done

bench:
	@echo "Running benchmarks..."
	cd savant_core/savant_core && cargo bench --no-default-features -- --show-output --nocapture


reformat:
	unify --in-place --recursive python
	unify --in-place --recursive savant_python/python
	black python
	black savant_python/python
	isort python
	isort savant_python/python

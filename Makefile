export PROJECT_DIR=$(CURDIR)
export PYTHON_VERSION=$(shell python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')

DS_NVBUF_DIR=$(PROJECT_DIR)/savant_deepstream/deepstream_nvbufsurface
DS_ENC_DIR=$(PROJECT_DIR)/savant_deepstream/deepstream_encoders
GST_DIR=$(PROJECT_DIR)/savant_gstreamer
SP_DIR=$(PROJECT_DIR)/savant_python

.PHONY: docs build_savant build_savant_release clean tests bench reformat \
        ds-nvbuf-dev ds-nvbuf-release ds-nvbuf-install \
        ds-nvbuf-test ds-nvbuf-pytest \
        ds-enc-dev ds-enc-release ds-enc-install \
        ds-enc-test ds-enc-pytest \
        gst-dev gst-release gst-install \
        gst-test gst-pytest \
        sp-dev sp-install sp-pytest \
        all-dev all-release \
        fmt clippy lint

dev: clean build_savant
release: clean build_savant_release

# -- formatting & linting ------------------------------------------------------

fmt:
	@echo "Running cargo fmt..."
	cargo fmt --all
	@echo "Running ruff format..."
	ruff format $(DS_NVBUF_DIR)/pytests $(DS_ENC_DIR)/pytests $(GST_DIR)/pytests \
	            $(SP_DIR)/pytests python/nvbufsurface 2>/dev/null || true

clippy:
	@echo "Running clippy on savant_gstreamer..."
	cd $(GST_DIR) && cargo clippy --all-targets -- -D warnings
	@echo "Running clippy on deepstream_nvbufsurface..."
	cd $(DS_NVBUF_DIR) && cargo clippy --all-targets -- -D warnings
	@echo "Running clippy on deepstream_encoders..."
	cd $(DS_ENC_DIR) && cargo clippy --all-targets -- -D warnings

lint: fmt clippy
	@echo "Running ruff check..."
	ruff check $(DS_NVBUF_DIR)/pytests $(DS_ENC_DIR)/pytests $(GST_DIR)/pytests \
	           $(SP_DIR)/pytests python/nvbufsurface --fix
	@echo "Lint complete."

# -- aggregate targets: build + test + install everything ---------------------

all-dev: fmt clippy lint \
         dev install \
         gst-dev gst-test gst-install gst-pytest \
         ds-nvbuf-dev ds-nvbuf-test ds-nvbuf-install ds-nvbuf-pytest \
         ds-enc-dev ds-enc-test ds-enc-install ds-enc-pytest

all-release: fmt clippy lint \
             release install \
             gst-release gst-test gst-install gst-pytest \
             ds-nvbuf-release ds-nvbuf-test ds-nvbuf-install ds-nvbuf-pytest \
             ds-enc-release ds-enc-test ds-enc-install ds-enc-pytest

# -----------------------------------------------------------------------------

install-with-optional-deps:
	@WHL_NAME=$$(ls $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl); \
	echo "Installing $$WHL_NAME[clientsdk]"; \
	pip install --force-reinstall "$$WHL_NAME[clientsdk]"; \
	echo "Installed $$WHL_NAME[clientsdk]"

install:
	@WHL_NAME=$$(ls $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl); \
	echo "Installing $$WHL_NAME"; \
	pip install --force-reinstall "$$WHL_NAME"; \
	echo "Installed $$WHL_NAME"

# -- deepstream_nvbufsurface Python bindings ----------------------------------

ds-nvbuf-dev:
	@echo "Building deepstream_nvbufsurface (dev)..."
	cd $(DS_NVBUF_DIR) && maturin build -f -o $(PROJECT_DIR)/dist

ds-nvbuf-release:
	@echo "Building deepstream_nvbufsurface (release)..."
	cd $(DS_NVBUF_DIR) && maturin build --release -f -o $(PROJECT_DIR)/dist

ds-nvbuf-install:
	@WHL_NAME=$$(ls -t $(PROJECT_DIR)/dist/deepstream_nvbufsurface*$(PYTHON_VERSION)*.whl | head -1); \
	echo "Installing $$WHL_NAME"; \
	pip install --force-reinstall "$$WHL_NAME"; \
	echo "Installed $$WHL_NAME"

ds-nvbuf-test:
	@echo "Running deepstream_nvbufsurface Rust tests..."
	cd $(DS_NVBUF_DIR) && cargo test -- --test-threads=1

ds-nvbuf-pytest: ds-nvbuf-dev ds-nvbuf-install
	@echo "Running deepstream_nvbufsurface Python tests..."
	cd $(DS_NVBUF_DIR) && python3 -m pytest pytests/ -v --tb=short

# -- deepstream_encoders Python bindings --------------------------------------

ds-enc-dev:
	@echo "Building deepstream_encoders (dev)..."
	cd $(DS_ENC_DIR) && maturin build -f -o $(PROJECT_DIR)/dist

ds-enc-release:
	@echo "Building deepstream_encoders (release)..."
	cd $(DS_ENC_DIR) && maturin build --release -f -o $(PROJECT_DIR)/dist

ds-enc-install:
	@WHL_NAME=$$(ls -t $(PROJECT_DIR)/dist/deepstream_encoders*$(PYTHON_VERSION)*.whl | head -1); \
	echo "Installing $$WHL_NAME"; \
	pip install --force-reinstall "$$WHL_NAME"; \
	echo "Installed $$WHL_NAME"

ds-enc-test:
	@echo "Running deepstream_encoders Rust tests..."
	cd $(DS_ENC_DIR) && cargo test -- --test-threads=1

ds-enc-pytest: ds-enc-dev ds-enc-install
	@echo "Running deepstream_encoders Python tests..."
	cd $(DS_ENC_DIR) && python3 -m pytest pytests/ -v --tb=short

# -- savant_gstreamer Python bindings -----------------------------------------

gst-dev:
	@echo "Building savant_gstreamer (dev)..."
	cd $(GST_DIR) && maturin build -f -o $(PROJECT_DIR)/dist

gst-release:
	@echo "Building savant_gstreamer (release)..."
	cd $(GST_DIR) && maturin build --release -f -o $(PROJECT_DIR)/dist

gst-install:
	@WHL_NAME=$$(ls -t $(PROJECT_DIR)/dist/savant_gstreamer*$(PYTHON_VERSION)*.whl | head -1); \
	echo "Installing $$WHL_NAME"; \
	pip install --force-reinstall "$$WHL_NAME"; \
	echo "Installed $$WHL_NAME"

gst-test:
	@echo "Running savant_gstreamer Rust tests..."
	cd $(GST_DIR) && cargo test -- --test-threads=1

gst-pytest: gst-dev gst-install
	@echo "Running savant_gstreamer Python tests..."
	cd $(GST_DIR) && python3 -m pytest pytests/ -v --tb=short

# -- savant_python Python bindings --------------------------------------------

sp-dev:
	@echo "Building savant_python (dev)..."
	cd $(SP_DIR) && maturin build -f -o $(PROJECT_DIR)/dist

sp-install:
	@WHL_NAME=$$(ls -t $(PROJECT_DIR)/dist/savant_rs*$(PYTHON_VERSION)*.whl | head -1); \
	echo "Installing $$WHL_NAME"; \
	pip install --force-reinstall "$$WHL_NAME"; \
	echo "Installed $$WHL_NAME"

sp-pytest: sp-dev sp-install
	@echo "Running savant_python Python tests..."
	cd $(SP_DIR) && python3 -m pytest pytests/ -v --tb=short

docs: dev install gst-dev gst-install ds-nvbuf-dev ds-nvbuf-install ds-enc-dev ds-enc-install
	@echo "Building docs..."
	cd $(PROJECT_DIR)/docs && LC_ALL=C.utf8 PATH="$(PROJECT_DIR)/venv/bin:$$PATH" make clean html
	tar --dereference --hard-dereference --directory $(PROJECT_DIR)/docs/build/html -cvf $(PROJECT_DIR)/docs-artifact.tar .

build_savant:
	@echo "Building..."
	utils/build.sh debug

build_savant_release:
	@echo "Building..."
	utils/build.sh release

clean:
	@echo "Cleaning..."
	rm -rf $(PROJECT_DIR)/dist/*.whl

pythontests:
	@echo "Running tests..."
	cd savant_python && cargo build && cargo test --no-default-features -- --test-threads=1 # --show-output --nocapture

core-tests:
	@echo "Running core lib tests..."
	cd savant_core && cargo build && cargo test -- --test-threads=1 # --show-output --nocapture

bench:
	@echo "Running benchmarks..."
	cd savant_core && cargo bench --no-default-features -- --show-output --nocapture


reformat:
	unify --in-place --recursive python
	unify --in-place --recursive savant_python/python
	black python
	black savant_python/python
	isort python
	isort savant_python/python

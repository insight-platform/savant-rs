export PROJECT_DIR=$(CURDIR)
export PYTHON_VERSION=$(shell python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')

.PHONY: docs clippy build_savant build_savant_release clean tests bench

dev: clean clippy build_savant
release: clean clippy build_savant_release

install:
	pip install --force-reinstall $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl

docs: dev install docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

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
	cd savant_python && cargo build && cargo test --no-default-features -- --show-output --nocapture --test-threads=1

core-tests:
	@echo "Running core lib tests..."
	cd savant_core && cargo build && cargo test -- --show-output --nocapture --test-threads=1

bench:
	@echo "Running benchmarks..."
	cd savant_core && cargo bench --no-default-features -- --show-output --nocapture

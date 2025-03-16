export PROJECT_DIR=$(CURDIR)
export PYTHON_VERSION=$(shell python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')

.PHONY: docs build_savant build_savant_release clean tests bench reformat

dev: clean build_savant
release: clean build_savant_release

install:
	pip install --force-reinstall $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl

docs: dev install docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

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
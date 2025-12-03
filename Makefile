export PROJECT_DIR=$(CURDIR)
export PYTHON_VERSION=$(shell python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')

.PHONY: docs build_savant build_savant_release clean tests bench reformat

dev: clean build_savant
release: clean build_savant_release

install:
	@WHL_NAME=$$(ls $(PROJECT_DIR)/dist/*$(PYTHON_VERSION)*.whl); \
	echo "Installing $$WHL_NAME[clientsdk]"; \
	pip install --force-reinstall "$$WHL_NAME[clientsdk]"; \
	echo "Installed $$WHL_NAME[clientsdk]"

docs:
	@echo "Building docs..."
	make dev install
	cd $(PROJECT_DIR)/docs && make clean html
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

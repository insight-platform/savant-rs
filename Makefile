export PROJECT_DIR=$(CURDIR)

.PHONY: docs clippy build_savant build_savant_release clean tests bench

dev: export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib:$(CURDIR)/target/debug/deps:$(CARGO_TARGET_DIR)/debug/deps
dev: clean clippy build build_savant build_python_plugins

release: export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib:$(CURDIR)/target/release/deps:$(CARGO_TARGET_DIR)/release/deps
release: clean clippy build_release build_savant_release build_python_plugins_release

install:
	pip install --force-reinstall $(PROJECT_DIR)/dist/*.whl

docs: dev install docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

build:
	@echo "Building..."
	cargo build
	plugins/prepare_native_plugins.sh debug

build_release:
	@echo "Building..."
	cargo build --release
	plugins/prepare_native_plugins.sh release

build_savant:
	@echo "Building..."
	cd savant_python && CARGO_INCREMENTAL=true maturin build -o $(PROJECT_DIR)/dist

build_savant_release:
	@echo "Building..."
	cd savant_python && maturin build -f --release -o $(PROJECT_DIR)/dist

build_python_plugins:
	@echo "Building plugins..."
	for d in $(wildcard plugins/python/*); do \
		cd $$d && CARGO_INCREMENTAL=true maturin build -o $(PROJECT_DIR)/dist; \
		cd $(PROJECT_DIR); \
	done
	# cd plugins/python/savant_py_plugin_sample && CARGO_INCREMENTAL=true maturin build -o $(PROJECT_DIR)/dist

build_python_plugins_release:
	@echo "Building plugins..."
	for d in $(wildcard plugins/python/*); do \
		cd $$d && maturin build -f --release -o $(PROJECT_DIR)/dist; \
		cd $(PROJECT_DIR); \
	done
	# cd plugins/python/savant_py_plugin_sample && maturin build -f --release -o $(PROJECT_DIR)/dist

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

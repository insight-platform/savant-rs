

.PHONY: docs clippy build_savant build_savant_release clean tests bench

dev: export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib:$(CURDIR)/target/debug/deps:$(CARGO_TARGET_DIR)/debug/deps
dev: clean clippy build build_savant build_plugin

release: export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib:$(CURDIR)/target/release/deps:$(CARGO_TARGET_DIR)/release/deps
release: clean clippy build_release build_savant_release build_plugin_release

install:
	find . -name '*.whl' -exec pip install --force-reinstall {} \;

docs: dev docs/source/index.rst
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
	cd savant_python && CARGO_INCREMENTAL=true maturin build -o dist

build_savant_release:
	@echo "Building..."
	cd savant_python && maturin build -f --release -o dist

build_plugin:
	@echo "Building plugin..."
	cd plugins/python/savant_py_plugin_sample && CARGO_INCREMENTAL=true maturin build -o dist

build_plugin_release:
	@echo "Building plugin..."
	cd plugins/python/savant_py_plugin_sample && maturin build -f --release -o dist

clean:
	@echo "Cleaning..."
	find . -name '*.whl' -exec rm -rf {} \;

pythontests:
	@echo "Running tests..."
	cd savant_python && cargo build && cargo test --no-default-features -- --show-output --nocapture --test-threads=1

core-tests:
	@echo "Running core lib tests..."
	cd savant_core && cargo build && cargo test -- --show-output --nocapture --test-threads=1

bench:
	@echo "Running benchmarks..."
	cd savant_core && cargo bench --no-default-features -- --show-output --nocapture

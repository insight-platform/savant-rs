

.PHONY: docs clippy build_savant build_savant_release clean tests bench

dev: export LD_LIBRARY_PATH := $(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib:$(CURDIR)/target/debug
dev: clean clippy build_savant build_plugin

release: export LD_LIBRARY_PATH := $(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib:$(CURDIR)/target/release
release: clean clippy build_savant_release build_plugin_release

docs: dev docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

build_savant:
	@echo "Building..."
	cd savant_python && CARGO_INCREMENTAL=true maturin build -o dist && pip install --force-reinstall dist/*.whl

build_plugin:
	@echo "Building plugin..."
	cd savant_py_plugin_sample && CARGO_INCREMENTAL=true maturin build -o dist && pip install --force-reinstall dist/*.whl

build_savant_release:
	@echo "Building..."
	cd savant_python && maturin build --release -o dist && pip install --force-reinstall dist/*.whl

build_plugin_release:
	@echo "Building plugin..."
	cd savant_py_plugin_sample && maturin build --release -o dist && pip install --force-reinstall dist/*.whl

clean:
	@echo "Cleaning..."
	cd savant_python && rm -rf dist/*.whl

pythontests:
	@echo "Running tests..."
	cd savant_python && cargo build && cargo test --no-default-features -- --show-output --nocapture --test-threads=1

core-tests:
	@echo "Running core lib tests..."
	cd savant_core && cargo build && cargo test -- --show-output --nocapture --test-threads=1

bench:
	@echo "Running benchmarks..."
	cd savant_core && cargo bench --no-default-features -- --show-output --nocapture

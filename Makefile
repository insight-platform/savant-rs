dev: clean clippy build install
release: clean clippy tests build_release install

docs: build install docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

sample_plugin: sample_plugin/src/lib.rs sample_plugin/Cargo.toml
	@echo "Building sample plugin..."
	cd sample_plugin && cargo build

build:
	@echo "Building..."
	cd savant && CARGO_INCREMENTAL=true maturin build -o dist

build_release:
	@echo "Building..."
	cd savant && maturin build --release -o dist

install:
	@echo "Installing..."
	cd savant && pip3.10 install --force-reinstall dist/*.whl

clean:
	@echo "Cleaning..."
	cd savant && rm -rf dist/*.whl

tests: sample_plugin
	@echo "Running tests..."
	cd savant && cargo test --no-default-features  -- --nocapture

bench: sample_plugin
	@echo "Running benchmarks..."
	cd savant && cargo bench --no-default-features -- --nocapture

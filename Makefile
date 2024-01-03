.PHONY: docs clippy build_savant build_savant_release clean tests bench

dev: clean clippy build_savant

release: clean clippy build_savant_release

docs: build_savant docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

build_savant:
	@echo "Building..."
	cd savant_python && CARGO_INCREMENTAL=true maturin build -o dist && pip install --force-reinstall dist/*.whl

build_savant_release:
	@echo "Building..."
	cd savant_python && maturin build --release -o dist

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

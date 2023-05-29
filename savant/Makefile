dev: clean clippy tests build install

clippy:
	@echo "Running clippy..."
	cargo clippy

build:
	@echo "Building..."
	RUSTFLAGS=" -C target-cpu=native -C opt-level=3" maturin build --release -o dist -i python3.10

install:
	@echo "Installing..."
	pip3.10 install --force-reinstall dist/*.whl

clean:
	@echo "Cleaning..."
	rm -rf dist/*.whl

tests:
	@echo "Running tests..."
	cargo test --no-default-features  -- --nocapture

bench:
	@echo "Running benchmarks..."
	cargo bench --no-default-features -- --nocapture

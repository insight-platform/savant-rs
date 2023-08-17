.PHONY: docs clippy build_savant build_etcd_dynamic_state build_savant_release build_etcd_dynamic_state_release clean tests bench

dev: clean clippy build_savant build_etcd_dynamic_state

docs: build_savant build_etcd_dynamic_state docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

build_savant:
	@echo "Building..."
	cd savant_python && CARGO_INCREMENTAL=true maturin dev

build_etcd_dynamic_state:
	@echo "Building Etcd dynamic state..."
	cd savant_etcd_dynamic_state && CARGO_INCREMENTAL=true maturin dev

build_savant_release:
	@echo "Building..."
	cd savant_python && maturin build --release -o dist

build_etcd_dynamic_state_release:
	@echo "Building..."
	cd savant_etcd_dynamic_state && maturin build --release -o dist

clean:
	@echo "Cleaning..."
	cd savant_python && rm -rf dist/*.whl
	cd savant_etcd_dynamic_state && rm -rf dist/*.whl

pythontests:
	@echo "Running tests..."
	cd savant_python && cargo build && cargo test --no-default-features -- --show-output --nocapture --test-threads=1

core-tests:
	@echo "Running core lib tests..."
	cd savant_core && cargo build && cargo test -- --show-output --nocapture --test-threads=1

bench:
	@echo "Running benchmarks..."
	cd savant_core && cargo bench --no-default-features -- --show-output --nocapture

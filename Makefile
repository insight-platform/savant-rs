.PHONY: docs clippy build_savant build_etcd_dynamic_state build_savant_release build_etcd_dynamic_state_release install_savant install_etcd_dynamic_state clean tests bench

dev: clean clippy build_savant build_etcd_dynamic_state install_savant install_etcd_dynamic_state
release: clean clippy tests build_savant_release build_etcd_dynamic_state_release install_savant install_etcd_dynamic_state

docs: build_savant install_savant build_etcd_dynamic_state install_etcd_dynamic_state docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

build_savant:
	@echo "Building..."
	cd savant && CARGO_INCREMENTAL=true maturin build -o dist

build_etcd_dynamic_state:
	@echo "Building Etcd dynamic state..."
	cd savant_etcd_dynamic_state && CARGO_INCREMENTAL=true maturin build -o dist

build_savant_release:
	@echo "Building..."
	cd savant && maturin build --release -o dist

build_etcd_dynamic_state_release:
	@echo "Building..."
	cd savant_etcd_dynamic_state && maturin build --release -o dist

install_savant:
	@echo "Installing..."
	cd savant && pip3.10 install --force-reinstall dist/*.whl

install_etcd_dynamic_state:
	@echo "Installing..."
	cd savant_etcd_dynamic_state && pip3.10 install --force-reinstall dist/*.whl

clean:
	@echo "Cleaning..."
	cd savant && rm -rf dist/*.whl
	cd savant_etcd_dynamic_state && rm -rf dist/*.whl

tests:
	@echo "Running tests..."
	cd savant && cargo build && cargo test --no-default-features -- --show-output --nocapture --test-threads=1

core-tests:
	@echo "Running core lib tests..."
	cd savant_core && cargo build && cargo test -- --show-output --nocapture --test-threads=1

bench:
	@echo "Running benchmarks..."
	cd savant && cargo bench --no-default-features -- --show-output --nocapture

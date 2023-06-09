dev: clean clippy build_savant build_etcd_dynamic_state install_savant install_etcd_dynamic_state
release: clean clippy tests build_savant_release build_etcd_dynamic_state_release install_savant install_etcd_dynamic_state

docs: build_savant install_savant docs/source/index.rst
	@echo "Building docs..."
	cd docs && make clean html

clippy:
	@echo "Running clippy..."
	cargo clippy

sample_plugin: sample_plugin/src/lib.rs sample_plugin/Cargo.toml
	@echo "Building sample plugin..."
	cd sample_plugin && cargo build

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

tests: sample_plugin
	@echo "Running tests..."
	cd savant && cargo test --no-default-features  -- --nocapture

bench: sample_plugin
	@echo "Running benchmarks..."
	cd savant && cargo bench --no-default-features -- --nocapture

# Savant Primitives And Optimized Algorithms

Savant Library with new generation primitives re-implemented in Rust.

Run tests:

```bash
cargo test --no-default-features
```

Run benchmarks:
```bash
cargo bench --no-default-features
```

Build Wheel:

```bash
RUSTFLAGS=" -C target-cpu=x86-64-v3 -C opt-level=3" maturin build --release -o dist
```

Install Wheel:

```bash
pip3 install --force-reinstall dist/savant_primitives-0.1.2-cp38-cp38-manylinux2014_x86_64.whl
```

## License

[Apache License 2.0](LICENSE)
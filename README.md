# SavantPrimitives

Savant Library with new generation primitives re-implemented in Rust.

Run tests:

```bash
cargo test --no-default-features
```

Build Wheel:

```bash
RUSTFLAGS=" -C target-cpu=x86-64-v3 -C opt-level=3" maturin build --release --no-sdist -o dist
```

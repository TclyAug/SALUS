# SALUS: Large-Scale Homomorphic Circuit Synthesis via Logic-Aware LUT Optimization

This code is the implementation of the paper "SALUS: Large-Scale Homomorphic Circuit Synthesis via Logic-Aware LUT Optimization".

## Requirements

To build and run this project, you will need:

- `git`
- `make`
- `gcc` or `clang`
- `rust` and `cargo`
- `python3`


## Third-Party Dependencies

This repository uses two third-party dependencies under [`third_party`](third_party):

- [`third_party/abc`](third_party/abc): <https://github.com/berkeley-abc/abc>
- [`third_party/refined-tfhe-lhe`](third_party/refined-tfhe-lhe): <https://github.com/KAIST-CryptLab/refined-tfhe-lhe>

After cloning the repository, initialize the submodules with:

```bash
git submodule update --init --recursive
```

Build the vendored `ABC` binary with:

```bash
make -C third_party/abc
```

## Building SALUS

Build the Rust binaries with:

```bash
cargo build --release
```

## Generating a Homomorphic Circuit

```bash
cargo run --release --bin generate_circuit -- <input.{v|blif}> <output_dir> [--with-merge]
```
Options:

- `--with-merge`
  - enable LUT merging for `hybrid` / `multi` execution
- `--max-abc-iters N`
  - limit the number of repeated mapping iterations

Example: generate a preprocessed circuit from [`testcircuit/c7552.v`](testcircuit/c7552.v)

```bash
make -C third_party/abc
cargo run --release --bin generate_circuit -- testcircuit/c7552.v HomCircuit/c7552 --with-merge
```

## Executing a Preprocessed Circuit

```bash
cargo run --release --bin execute_circuit -- <preprocessed_dir> [input_bits] [--single-lut-mode|--hybrid-lut-mode|--multi-lut-mode] [--repeat-random N]
```

Execution modes:

- `--single-lut-mode`
  - only single-output HomLUT
- `--multi-lut-mode`
  - only multi-output HomLUT
- `--hybrid-lut-mode`
  - both single-output and multi-output HomLUT

Examples:

```bash
cargo run --release --bin execute_circuit -- HomCircuit/c7552 --hybrid-lut-mode --repeat-random 1
```

```bash
cargo run --release --bin execute_circuit -- HomCircuit/divisor --hybrid-lut-mode --repeat-random 1
```

## Example Workflow

```bash

cargo run --release --bin generate_circuit -- testcircuit/c7552.v HomCircuit/c7552 --with-merge
cargo run --release --bin execute_circuit -- HomCircuit/c7552 --hybrid-lut-mode --repeat-random 1
```

# raptorq
![CI](https://github.com/cberner/raptorq/actions/workflows/ci.yml/badge.svg)
[![Crates](https://img.shields.io/crates/v/raptorq.svg)](https://crates.io/crates/raptorq)
[![Documentation](https://docs.rs/raptorq/badge.svg)](https://docs.rs/raptorq)
[![PyPI](https://img.shields.io/pypi/v/raptorq.svg)](https://pypi.org/project/raptorq/)
[![License](https://img.shields.io/crates/l/raptorq)](https://crates.io/crates/raptorq)
[![dependency status](https://deps.rs/repo/github/cberner/raptorq/status.svg)](https://deps.rs/repo/github/cberner/raptorq)

### Overview

Rust implementation of RaptorQ (RFC6330)

Recovery properties:
Reconstruction probability after receiving K + h packets = 1 - 1/256^(h + 1). Where K is the number of packets in the
original message, and h is the number of additional packets received.
See "RaptorQ Technical Overview" by Qualcomm

### Examples
See the `examples/` directory for usage.

### Benchmarks

This clean-room implementation currently uses a generic solver for matrices up to 1120
intermediate symbols. An optimized large-matrix PI/inactivation solver is not implemented yet.

The following were run on a Ryzen 9 9950X3D @ 4.30GHz
```
Symbol size: 1280 bytes (without pre-built plan)
symbol count = 10, encoded 127 MB in 0.046secs, throughput: 22259.3Mbit/s
symbol count = 100, encoded 127 MB in 0.046secs, throughput: 22248.6Mbit/s
symbol count = 250, encoded 127 MB in 0.050secs, throughput: 20459.0Mbit/s
symbol count = 500, encoded 127 MB in 0.038secs, throughput: 26855.5Mbit/s

Symbol size: 1280 bytes (with pre-built plan)
symbol count = 10, encoded 127 MB in 0.063secs, throughput: 16252.8Mbit/s
symbol count = 100, encoded 127 MB in 0.034secs, throughput: 30101.1Mbit/s
symbol count = 250, encoded 127 MB in 0.051secs, throughput: 20057.8Mbit/s
symbol count = 500, encoded 127 MB in 0.038secs, throughput: 26855.5Mbit/s

Symbol size: 1280 bytes
symbol count = 10, decoded 127 MB in 0.174secs using 0.0% overhead, throughput: 5884.6Mbit/s
symbol count = 100, decoded 127 MB in 0.146secs using 0.0% overhead, throughput: 7009.8Mbit/s
symbol count = 250, decoded 127 MB in 0.146secs using 0.0% overhead, throughput: 7006.5Mbit/s
symbol count = 500, decoded 127 MB in 0.143secs using 0.0% overhead, throughput: 7136.4Mbit/s

symbol count = 10, decoded 127 MB in 0.187secs using 5.0% overhead, throughput: 5475.5Mbit/s
symbol count = 100, decoded 127 MB in 0.146secs using 5.0% overhead, throughput: 7009.8Mbit/s
symbol count = 250, decoded 127 MB in 0.140secs using 5.0% overhead, throughput: 7306.8Mbit/s
symbol count = 500, decoded 127 MB in 0.117secs using 5.0% overhead, throughput: 8722.3Mbit/s
```

The following were run on a Raspberry Pi 3 B+ (Cortex-A53 @ 1.4GHz), as of raptorq version 2.0.0

```
Symbol size: 1280 bytes (without pre-built plan)
symbol count = 10, encoded 127 MB in 5.078secs, throughput: 201.6Mbit/s
symbol count = 100, encoded 127 MB in 3.966secs, throughput: 258.1Mbit/s
symbol count = 250, encoded 127 MB in 4.293secs, throughput: 238.3Mbit/s
symbol count = 500, encoded 127 MB in 4.451secs, throughput: 229.3Mbit/s

Symbol size: 1280 bytes (with pre-built plan)
symbol count = 10, encoded 127 MB in 3.438secs, throughput: 297.8Mbit/s
symbol count = 100, encoded 127 MB in 2.476secs, throughput: 413.3Mbit/s
symbol count = 250, encoded 127 MB in 2.908secs, throughput: 351.8Mbit/s
symbol count = 500, encoded 127 MB in 3.085secs, throughput: 330.8Mbit/s

Symbol size: 1280 bytes
symbol count = 10, decoded 127 MB in 6.561secs using 0.0% overhead, throughput: 156.1Mbit/s
symbol count = 100, decoded 127 MB in 4.936secs using 0.0% overhead, throughput: 207.3Mbit/s
symbol count = 250, decoded 127 MB in 5.206secs using 0.0% overhead, throughput: 196.5Mbit/s
symbol count = 500, decoded 127 MB in 5.298secs using 0.0% overhead, throughput: 192.6Mbit/s

symbol count = 10, decoded 127 MB in 6.157secs using 5.0% overhead, throughput: 166.3Mbit/s
symbol count = 100, decoded 127 MB in 4.842secs using 5.0% overhead, throughput: 211.4Mbit/s
symbol count = 250, decoded 127 MB in 5.213secs using 5.0% overhead, throughput: 196.2Mbit/s
symbol count = 500, decoded 127 MB in 5.328secs using 5.0% overhead, throughput: 191.5Mbit/s
```

### Public API
Note that the additional classes exported by the `benchmarking` feature flag are not considered part of this
crate's public API. Breaking changes to those classes may occur without warning. The flag is only provided
so that internal classes can be used in this crate's benchmarks.

## Python bindings

The Python bindings are generated using [pyo3](https://github.com/PyO3/pyo3). 

Some operating systems require additional packages to be installed.
```
$ sudo apt install python3-dev cargo
```

[maturin](https://github.com/PyO3/maturin) is recommended for building the Python bindings in this crate.
```
$ pip install maturin
$ maturin build --features python
```

Alternatively, refer to the [Building and Distribution section](https://pyo3.rs/v0.8.5/building_and_distribution.html) in the [pyo3 user guide](https://pyo3.rs/v0.8.5/).
Note, you must pass the `--cargo-extra-args="--features python"` argument to Maturin when building this crate
to enable the Python binding features.

## License

Licensed under

 * Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be licensed as above, without any
additional terms or conditions.

# float16

[![Crates.io](https://img.shields.io/crates/v/float16.svg)](https://crates.io/crates/float16/) [![Documentation](https://docs.rs/float16/badge.svg)](https://docs.rs/float16/) ![Crates.io](https://img.shields.io/crates/l/half) [![Build status](https://github.com/Alexhuszagh/float16/actions/workflows/rust.yml/badge.svg?branch=main&event=push)](https://github.com/Alexhuszagh/float16/actions/workflows/rust.yml)

This crate implements a half-precision floating point `f16` type for Rust implementing the IEEE 754-2008 standard [`binary16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) a.k.a "half" format, as well as a `bf16` type implementing the [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) format.

This is a minimal variant which removes some features and introduce many build dependencies. It also allows the crate to run on much older versions of rustc (1.60+) while retaining the performance, still using intrinsics when available (Rust 1.63+ on ARM64 and Rust 1.68+ on x86/x86_64). If you would like to use this, we recommend converting to or from the analogous types in [half](https://crates.io/crates/half), which this crate was based off of.

## Usage

The `f16` and `bf16` types attempt to match existing Rust floating point type functionality where possible, and provides both conversion operations (such as to/from `f32` and `f64`) and basic arithmetic operations. Hardware support for these operations will be used whenever hardware support is available—either through instrinsics or targeted assembly—although a nightly Rust toolchain may be required for some hardware.

This crate provides [`no_std`](https://rust-embedded.github.io/book/intro/no-std.html) support so can easily be used in embedded code where a smaller float format is most useful.

*Requires Rust 1.60 or greater.*

See the [crate documentation](https://docs.rs/float16/) for more details.

### Hardware support

The following list details hardware support for floating point types in this crate. When using `std`
library, runtime CPU target detection will be used. To get the most performance benefits, compile
for specific CPU features which avoids the runtime overhead and works in a `no_std` environment.

| Architecture | CPU Target Feature | Notes |
| ------------ | ------------------ | ----- |
| `x86`/`x86_64` | `f16c` | This supports conversion to/from `f16` only (including vector SIMD) and does not support any `bf16` or arithmetic operations. |
| `aarch64` | `fp16` | This supports all operations on `f16` only. |

### More Documentation

- [Crate API Reference](https://docs.rs/float16/)
- [Latest Changes](CHANGELOG.md)

## License

This library is distributed under the terms of either of:

- [MIT License](LICENSES/MIT.txt) ([http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
- [Apache License, Version 2.0](LICENSES/Apache-2.0.txt) ([http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option.

This project is [REUSE-compliant](https://reuse.software/spec/). Copyrights are retained by their contributors. Some files may include explicit copyright notices and/or license [SPDX identifiers](https://spdx.dev/ids/). For full authorship information, see the version control history.

### Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

# half: `f16` type for Rust 
[![Crates.io](https://img.shields.io/crates/v/half.svg)](https://crates.io/crates/half/) [![Build Status](https://travis-ci.org/starkat99/half-rs.svg?branch=master)](https://travis-ci.org/starkat99/half-rs) [![Build status](https://ci.appveyor.com/api/projects/status/bi18aypi3h5r88gs?svg=true)](https://ci.appveyor.com/project/starkat99/half-rs)

This crate implements a half-precision floating point `f16` type for Rust implementing the IEEE 754-2008 `binary16` type.

## How-to Use

The `f16` type provides all the same operations as a normal Rust float type, but since it is primarily leveraged for
minimal floating point storage and no major hardware implements them, all math operations are done as an `f32` type.
`f16` by default provides `no_std` support so can easily be used in embedded code where a smaller float is most useful.

See the [crate documentation](https://docs.rs/half/) for more details.

### Optional Features

- **`serialize`** - Implement `Serialize` and `Deserialize` traits for `f16`. This adds a dependency on the `serde` 
crate. *Requires Rust >= 1.15.*

- **`use-intrinsics`** - Use hardware intrinsics for `f16` conversions if available on the compiler host target. By 
default, without this feature, conversions are done only in software, which will be the fallback if the host target does
not have hardware support. **Available only on Rust nightly channel.**

- **`std`** - Use Rust `std` library. Currently no additional functionality is enabled by this feature. Provided only
for forward-compatibility.

### More Documentation

- [Crate API Reference](https://docs.rs/half/)
- [Latest Changes](CHANGELOG.md)

## License

This library is distributed under the terms of either of:

* MIT license ([LICENSE-MIT](LICENSE-MIT) or
[http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option.

### Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
# Changelog

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.1] - 2019-10-04 <a name="1.3.1"></a>
### Fixed
- Corrected values of constants `EPSILON`, `MAX_10_EXP`, `MAX_EXP`, `MIN_10_EXP`, and `MIN_EXP`
  in `consts` module, as well as setting `consts::NAN` to match value of `f32::NAN` convertd to
  `f16`. By [@tspiteri].

## [1.3.0] - 2018-10-02 <a name="1.3.0"></a>
### Added
- `slice::from_bits_mut` and `slice::to_bits_mut` for conversion between mutable `u16` and `f16`
slices. Fixes [#16], by [@johannesvollmer].

## [1.2.0] - 2018-09-03 <a name="1.2.0"></a>
### Added
- `slice` and optional `vec` (only included with `std` feature) modules for conversions between `u16` and `f16`
buffers. Fixes [#14], by [@johannesvollmer].
- `to_bits` added to replace `as_bits`. Fixes [#12], by [@tspiteri].
### Fixed
- `serde` optional dependency no longer uses its default `std` feature.
### Deprecated
- `as_bits` has been deprecated; use `to_bits` instead.
- `serialize` cargo feature is deprecated; use `serde` instead.

## [1.1.2] - 2018-07-12 <a name="1.1.2"></a>
### Fixed
- Fixed compilation error in 1.1.1 on rustc < 1.27, now compiles again on rustc >= 1.10. Fixes [#11].

## [1.1.1] - 2018-06-24 - **Yanked** <a name="1.1.1"></a>
### ***Yanked***
*Not recommended due to introducing compilation error on rustc versions prior to 1.27.*
### Fixed
- Fix subnormal float conversions when `use-intrinsics` is not enabled. By [@Moongoodboy-K].

## [1.1.0] - 2018-03-17 <a name="1.1.0"></a>
### Added
- Made `to_f32` and `to_f64` public. Fixes [#7], by [@PSeitz].

## [1.0.2] - 2018-01-12 <a name="1.0.2"></a>
### Changed
- Update behavior of `is_sign_positive` and `is_sign_negative` to match the IEEE754 conforming
behavior of the standard library since Rust 1.20.0. Fixes [#3], by [@tspiteri].
- Small optimization on `is_nan` and `is_infinite` from [@tspiteri].
### Fixed
- Fix comparisons of +0 to -0 and comparisons involving negative numbers. Fixes [#2], by [@tspiteri].
- Fix loss of sign when converting `f16` and `f32` to `f16`, and case where `f64` NaN could be
converted to `f16` infinity instead of NaN. Fixes [#5], by [@tspiteri].

## [1.0.1] - 2017-08-30 <a name="1.0.1"></a>
### Added
- More README documentation.
- Badges and categories in crate metadata.
### Changed
- `serde` dependency updated to 1.0 stable.
- Writing changelog manually.

## [1.0.0] - 2017-02-03 <a name="1.0.0"></a>
### Added
- Update to `serde` 0.9 and stable Rust 1.15 for `serialize` feature.

## [0.1.1] - 2017-01-08 <a name="0.1.1"></a>
### Added
- Add `serde` support under new `serialize` feature.
### Changed
- Use `no_std` for crate by default.

## 0.1.0 - 2016-03-17 <a name="0.1.0"></a>
### Added
- Initial release of `f16` type.

[#2]: https://github.com/starkat99/half-rs/issues/2
[#3]: https://github.com/starkat99/half-rs/issues/3
[#5]: https://github.com/starkat99/half-rs/issues/5
[#7]: https://github.com/starkat99/half-rs/issues/7
[#11]: https://github.com/starkat99/half-rs/issues/11
[#12]: https://github.com/starkat99/half-rs/issues/12
[#14]: https://github.com/starkat99/half-rs/issues/14
[#16]: https://github.com/starkat99/half-rs/issues/16

[@tspiteri]: https://github.com/tspiteri
[@PSeitz]: https://github.com/PSeitz
[@Moongoodboy-K]: https://github.com/Moongoodboy-K
[@johannesvollmer]: https://github.com/johannesvollmer

[Unreleased]: https://github.com/starkat99/half-rs/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/starkat99/half-rs/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/starkat99/half-rs/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/starkat99/half-rs/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/starkat99/half-rs/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/starkat99/half-rs/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/starkat99/half-rs/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/starkat99/half-rs/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/starkat99/half-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/starkat99/half-rs/compare/v0.1.1...v1.0.0
[0.1.1]: https://github.com/starkat99/half-rs/compare/v0.1.0...v0.1.1

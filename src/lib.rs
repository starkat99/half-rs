//! A crate that provides support for half-precision 16-bit floating point
//! types.
//!
//! This crate provides the [`struct@f16`] type, which is an implementation of
//! the IEEE 754-2008 standard [`binary16`] a.k.a "half" floating point type.
//! This 16-bit floating point type is intended for efficient storage where the
//! full range and precision of a larger floating point value is not required.
//! This is especially useful for image storage formats.
//!
//! This crate also provides a [`struct@bf16`] type, an alternative 16-bit
//! floating point format. The [`bfloat16`] format is a truncated IEEE 754
//! standard `binary32` float that preserves the exponent to allow the same
//! range as [`f32`] but with only 8 bits of precision (instead of 11 bits for
//! [`struct@f16`]). See the [`struct@bf16`] type for details.
//!
//! Because [`struct@f16`] and [`struct@bf16`] are primarily for efficient
//! storage, floating point operations such as addition, multiplication, etc.
//! are not always implemented by hardware. When hardware does not support these
//! operations, this crate emulates them by converting the value to [`f32`]
//! before performing the operation and then back afterward.
//!
//! Note that conversion from [`f32`]/[`f64`] to both [`struct@f16`] and
//! [`struct@bf16`] are lossy operations, and just as converting a [`f64`] to
//! [`f32`] is lossy and does not have `Into`/`From` trait implementations, so
//! too do these smaller types not have those trait implementations either.
//! Instead, use `from_f32`/`from_f64` functions for the types in this crate. If
//! you don't care about lossy conversions and need trait conversions, use the
//! appropriate [`num-traits`] traits that are implemented.
//!
//! The crate supports `#[no_std]` when the `std` cargo feature is not enabled,
//! so can be used in embedded environments without using the Rust [`std`]
//! library. The `std` feature enables support for the standard library and is
//! enabled by default, see the [Cargo Features](#cargo-features) section below.
//!
//! # Hardware support
//!
//! Hardware support for these conversions and arithmetic will be used
//! whenever hardware support is available—either through instrinsics or
//! targeted assembly—although a nightly Rust toolchain may be required for some
//! hardware. When hardware supports it the functions and traits
//! [`HalfBitsSliceExt`] and [`HalfFloatSliceExt`] are used it
//! will also use vectorized SIMD intructions for increased efficiency.
//!
//! The following list details hardware support for floating point types in this
//! crate. When using `std` cargo feature, runtime CPU target detection will be
//! used. To get the most performance benefits, compile for specific CPU
//! features which avoids the runtime overhead and works in a
//! `no_std` environment.
//!
//! | Architecture | CPU Target Feature | Notes |
//! | ------------ | ------------------ | ----- |
//! | `x86`/`x86_64` | `f16c` | This supports conversion to/from [`struct@f16`] only (including vector SIMD) and does not support any [`struct@bf16`] or arithmetic operations. |
//! | `aarch64` | `fp16` | This supports all operations on [`struct@f16`] only. |
//!
//! # Cargo Features
//!
//! To support numerous features, use the [float16-ext] package, which
//! implements its own `f16` and `bf16` types that support features like
//! `serde` serializing, zero-copy logic, and more.
//!
//! [`std`]: https://doc.rust-lang.org/std/
//! [`binary16`]: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
//! [`bfloat16`]: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
#![allow(unknown_lints)]
#![allow(clippy::verbose_bit_mask, clippy::cast_lossless)]
#![cfg_attr(not(feature = "std"), no_std)]
#![doc(html_root_url = "https://docs.rs/float16/0.1.0")]
#![doc(test(attr(deny(warnings), allow(unused))))]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod bfloat;
mod binary16;
mod leading_zeros;
mod slice;

pub use bfloat::bf16;
pub use binary16::f16;

#[cfg(not(target_arch = "spirv"))]
pub use crate::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

// Keep this module private to crate
mod private {
    use crate::{bf16, f16};

    pub trait SealedHalf {}

    impl SealedHalf for f16 {
    }
    impl SealedHalf for bf16 {
    }
}

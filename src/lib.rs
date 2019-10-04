//! A crate that provides support for the half-precision floating point type.
//!
//! This crate provides the `f16` type, which is an implementation of the IEEE 754-2008 `binary16`
//! floating point type. This 'half' precision floating point type is intended for efficient storage
//! where the full range and precision of a larger floating point value is not required. This is
//! especially useful for image storage formats.
//!
//! Because `f16` is primarily for efficient storage, floating point operations are not implemented.
//! Operations should be performed with `f32` or higher-precision types and converted to/from `f16`
//! as necessary.
//!
//! Some hardware architectures provide support for 16-bit floating point conversions. Enable the
//! `use-intrinsics` feature to use LLVM intrinsics for hardware conversions. This crate does no
//! checks on whether the hardware supports the feature. This feature currently only works on
//! nightly Rust due to a compiler feature gate.
//!
//! Support for `serde` crate `Serialize` and `Deserialize` traits is provided when the `serde`
//! feature is enabled. This adds a dependency on `serde` crate so is an optional feature that works
//! on Rust 1.15 or newer.
//!
//! The crate uses `#[no_std]` by default, so can be used in embedded environments without using the
//! Rust `std` library. A `std` feature is available, which enables additional utilities using the
//! `std` library, such as the `vec` module that provides zero-copy `Vec` conversions.

#![warn(
    missing_docs, missing_copy_implementations, missing_debug_implementations, trivial_casts,
    trivial_numeric_casts, unused_extern_crates, unused_import_braces
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "use-intrinsics", feature(link_llvm_intrinsics))]

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

#[cfg(feature = "std")]
extern crate core;

use core::cmp::Ordering;
use core::fmt::{Debug, Display, Error, Formatter, LowerExp, UpperExp};
use core::num::{FpCategory, ParseFloatError};
use core::str::FromStr;

/// The 16-bit floating point type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct f16(u16);

pub mod consts {
    //! Useful `f16` constants.

    use super::f16;

    /// 16-bit equivalent of `std::f32::DIGITS`
    pub const DIGITS: u32 = 3;
    /// 16-bit floating point epsilon. `9.7656e-4`
    pub const EPSILON: f16 = f16(0x1400u16);
    /// 16-bit positive infinity.
    pub const INFINITY: f16 = f16(0x7C00u16);
    /// 16-bit equivalent of `std::f32::MANTISSA_DIGITS`
    pub const MANTISSA_DIGITS: u32 = 11;
    /// Largest finite `f16` value. `65504`
    pub const MAX: f16 = f16(0x7BFF);
    /// 16-bit equivalent of `std::f32::MAX_10_EXP`
    pub const MAX_10_EXP: i32 = 4;
    /// 16-bit equivalent of `std::f32::MAX_EXP`
    pub const MAX_EXP: i32 = 16;
    /// Smallest finite `f16` value. `-65504`
    pub const MIN: f16 = f16(0xFBFF);
    /// 16-bit equivalent of `std::f32::MIN_10_EXP`
    pub const MIN_10_EXP: i32 = -4;
    /// 16-bit equivalent of `std::f32::MIN_EXP`
    pub const MIN_EXP: i32 = -13;
    /// Smallest positive, normalized `f16` value. Approx. `6.10352e−5`
    pub const MIN_POSITIVE: f16 = f16(0x0400u16);
    /// 16-bit NaN.
    pub const NAN: f16 = f16(0x7E00u16);
    /// 16-bit negative infinity.
    pub const NEG_INFINITY: f16 = f16(0xFC00u16);
    /// 16-bit equivalent of `std::f32::RADIX`
    pub const RADIX: u32 = 2;

    /// 16-bit minimum positive subnormal value. Approx. `5.96046e−8`
    pub const MIN_POSITIVE_SUBNORMAL: f16 = f16(0x0001u16);
    /// 16-bit maximum subnormal value. Approx. `6.09756e−5`
    pub const MAX_SUBNORMAL: f16 = f16(0x03FFu16);

    /// 16-bit floating point `1.0`
    pub const ONE: f16 = f16(0x3C00u16);
    /// 16-bit floating point `0.0`
    pub const ZERO: f16 = f16(0x0000u16);
    /// 16-bit floating point `-0.0`
    pub const NEG_ZERO: f16 = f16(0x8000u16);

    /// Euler's number.
    pub const E: f16 = f16(0x4170u16);
    /// Archimedes' constant.
    pub const PI: f16 = f16(0x4248u16);
    /// 1.0/pi
    pub const FRAC_1_PI: f16 = f16(0x3518u16);
    /// 1.0/sqrt(2.0)
    pub const FRAC_1_SQRT_2: f16 = f16(0x39A8u16);
    /// 2.0/pi
    pub const FRAC_2_PI: f16 = f16(0x3918u16);
    /// 2.0/sqrt(pi)
    pub const FRAC_2_SQRT_PI: f16 = f16(0x3C83u16);
    /// pi/2.0
    pub const FRAC_PI_2: f16 = f16(0x3E48u16);
    /// pi/3.0
    pub const FRAC_PI_3: f16 = f16(0x3C30u16);
    /// pi/4.0
    pub const FRAC_PI_4: f16 = f16(0x3A48u16);
    /// pi/6.0
    pub const FRAC_PI_6: f16 = f16(0x3830u16);
    /// pi/8.0
    pub const FRAC_PI_8: f16 = f16(0x3648u16);
    /// ln(10.0)
    pub const LN_10: f16 = f16(0x409Bu16);
    /// ln(2.0)
    pub const LN_2: f16 = f16(0x398Cu16);
    /// log10(e)
    pub const LOG10_E: f16 = f16(0x36F3u16);
    /// log2(e)
    pub const LOG2_E: f16 = f16(0x3DC5u16);
    /// sqrt(2)
    pub const SQRT_2: f16 = f16(0x3DA8u16);
}

impl f16 {
    /// Constructs a 16-bit floating point value from the raw bits.
    #[inline]
    pub fn from_bits(bits: u16) -> f16 {
        f16(bits)
    }

    /// Constructs a 16-bit floating point value from a 32-bit floating point value.
    ///
    /// If the 32-bit value is to large to fit in 16-bits, +/- infinity will result. NaN values are
    /// preserved. 32-bit subnormal values are too tiny to be represented in 16-bits and result in
    /// +/- 0. Exponents that underflow the minimum 16-bit exponent will result in 16-bit subnormals
    /// or +/- 0. All other values are truncated and rounded to the nearest representable 16-bit
    /// value.
    #[inline]
    pub fn from_f32(value: f32) -> f16 {
        f16(convert::f32_to_f16(value))
    }

    /// Constructs a 16-bit floating point value from a 64-bit floating point value.
    ///
    /// If the 64-bit value is to large to fit in 16-bits, +/- infinity will result. NaN values are
    /// preserved. 64-bit subnormal values are too tiny to be represented in 16-bits and result in
    /// +/- 0. Exponents that underflow the minimum 16-bit exponent will result in 16-bit subnormals
    /// or +/- 0. All other values are truncated and rounded to the nearest representable 16-bit
    /// value.
    #[inline]
    pub fn from_f64(value: f64) -> f16 {
        f16(convert::f64_to_f16(value))
    }

    /// Converts an `f16` into the underlying bit representation.
    #[inline]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f16` into the underlying bit representation.
    #[deprecated(since = "1.2.0", note = "renamed to to_bits")]
    #[inline]
    pub fn as_bits(self) -> u16 {
        self.to_bits()
    }

    /// Converts an `f16` value in a `f32` value.
    ///
    /// This conversion is lossless as all 16-bit floating point values can be represented exactly
    /// in 32-bit floating point.
    #[inline]
    pub fn to_f32(self) -> f32 {
        convert::f16_to_f32(self.0)
    }

    /// Converts an `f16` value in a `f64` value.
    ///
    /// This conversion is lossless as all 16-bit floating point values can be represented exactly
    /// in 64-bit floating point.
    #[inline]
    pub fn to_f64(self) -> f64 {
        convert::f16_to_f64(self.0)
    }

    /// Returns `true` if this value is `NaN` and `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let nan = half::consts::NAN;
    /// let f = f16::from_f32(7.0_f32);
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[inline]
    pub fn is_nan(self) -> bool {
        self.0 & 0x7FFFu16 > 0x7C00u16
    }

    /// Returns `true` if this value is positive infinity or negative infinity and `false`
    /// otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let f = f16::from_f32(7.0f32);
    /// let inf = half::consts::INFINITY;
    /// let neg_inf = half::consts::NEG_INFINITY;
    /// let nan = half::consts::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.0 & 0x7FFFu16 == 0x7C00u16
    }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let f = f16::from_f32(7.0f32);
    /// let inf = half::consts::INFINITY;
    /// let neg_inf = half::consts::NEG_INFINITY;
    /// let nan = half::consts::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0 & 0x7C00u16 != 0x7C00u16
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal, or `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let min = half::consts::MIN_POSITIVE;
    /// let max = half::consts::MAX;
    /// let lower_than_min = f16::from_f32(1.0e-10_f32);
    /// let zero = f16::from_f32(0.0_f32);
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!half::consts::NAN.is_normal());
    /// assert!(!half::consts::INFINITY.is_normal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    #[inline]
    pub fn is_normal(self) -> bool {
        let exp = self.0 & 0x7C00u16;
        exp != 0x7C00u16 && exp != 0
    }

    /// Returns the floating point category of the number.
    ///
    /// If only one property is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::FpCategory;
    /// use half::f16;
    ///
    /// let num = f16::from_f32(12.4_f32);
    /// let inf = half::consts::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    pub fn classify(self) -> FpCategory {
        let exp = self.0 & 0x7C00u16;
        let man = self.0 & 0x03FFu16;
        if exp == 0 {
            if man == 0 {
                FpCategory::Zero
            } else {
                FpCategory::Subnormal
            }
        } else if exp == 0x7C00u16 {
            if man == 0 {
                FpCategory::Infinite
            } else {
                FpCategory::Nan
            }
        } else {
            FpCategory::Normal
        }
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// * `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// * `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// * `NAN` if the number is `NAN`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let f = f16::from_f32(3.5_f32);
    ///
    /// assert_eq!(f.signum(), f16::from_f32(1.0));
    /// assert_eq!(half::consts::NEG_INFINITY.signum(), f16::from_f32(-1.0));
    ///
    /// assert!(half::consts::NAN.signum().is_nan());
    /// ```
    pub fn signum(self) -> f16 {
        if self.is_nan() {
            self
        } else if self.0 & 0x8000u16 != 0 {
            f16::from_f32(-1.0)
        } else {
            f16::from_f32(1.0)
        }
    }

    /// Returns `true` if and only if `self` has a positive sign, including `+0.0`, `NaNs` with
    /// positive sign bit and positive infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let nan = half::consts::NAN;
    /// let f = f16::from_f32(7.0_f32);
    /// let g = f16::from_f32(-7.0_f32);
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// // `NaN` can be either positive or negative
    /// assert!(nan.is_sign_positive() != nan.is_sign_negative());
    /// ```
    #[inline]
    pub fn is_sign_positive(self) -> bool {
        self.0 & 0x8000u16 == 0
    }

    /// Returns `true` if and only if `self` has a negative sign, including `-0.0`, `NaNs` with
    /// negative sign bit and negative infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::f16;
    ///
    /// let nan = half::consts::NAN;
    /// let f = f16::from_f32(7.0f32);
    /// let g = f16::from_f32(-7.0f32);
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// // `NaN` can be either positive or negative
    /// assert!(nan.is_sign_positive() != nan.is_sign_negative());
    /// ```
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        self.0 & 0x8000u16 != 0
    }
}

impl From<f16> for f32 {
    fn from(x: f16) -> f32 {
        x.to_f32()
    }
}

impl From<f16> for f64 {
    fn from(x: f16) -> f64 {
        x.to_f64()
    }
}

impl From<i8> for f16 {
    fn from(x: i8) -> f16 {
        // Convert to f32, then to f16
        f16::from_f32(f32::from(x))
    }
}

impl From<u8> for f16 {
    fn from(x: u8) -> f16 {
        // Convert to f32, then to f16
        f16::from_f32(f32::from(x))
    }
}

impl PartialEq for f16 {
    fn eq(&self, other: &f16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            (self.0 == other.0) || ((self.0 | other.0) & 0x7FFFu16 == 0)
        }
    }
}

impl PartialOrd for f16 {
    fn partial_cmp(&self, other: &f16) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else {
            let neg = self.0 & 0x8000u16 != 0;
            let other_neg = other.0 & 0x8000u16 != 0;
            match (neg, other_neg) {
                (false, false) => Some(self.0.cmp(&other.0)),
                (false, true) => {
                    if (self.0 | other.0) & 0x7FFFu16 == 0 {
                        Some(Ordering::Equal)
                    } else {
                        Some(Ordering::Greater)
                    }
                }
                (true, false) => {
                    if (self.0 | other.0) & 0x7FFFu16 == 0 {
                        Some(Ordering::Equal)
                    } else {
                        Some(Ordering::Less)
                    }
                }
                (true, true) => Some(other.0.cmp(&self.0)),
            }
        }
    }

    fn lt(&self, other: &f16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            let neg = self.0 & 0x8000u16 != 0;
            let other_neg = other.0 & 0x8000u16 != 0;
            match (neg, other_neg) {
                (false, false) => self.0 < other.0,
                (false, true) => false,
                (true, false) => (self.0 | other.0) & 0x7FFFu16 != 0,
                (true, true) => self.0 > other.0,
            }
        }
    }

    fn le(&self, other: &f16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            let neg = self.0 & 0x8000u16 != 0;
            let other_neg = other.0 & 0x8000u16 != 0;
            match (neg, other_neg) {
                (false, false) => self.0 <= other.0,
                (false, true) => (self.0 | other.0) & 0x7FFFu16 == 0,
                (true, false) => true,
                (true, true) => self.0 >= other.0,
            }
        }
    }

    fn gt(&self, other: &f16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            let neg = self.0 & 0x8000u16 != 0;
            let other_neg = other.0 & 0x8000u16 != 0;
            match (neg, other_neg) {
                (false, false) => self.0 > other.0,
                (false, true) => (self.0 | other.0) & 0x7FFFu16 != 0,
                (true, false) => false,
                (true, true) => self.0 < other.0,
            }
        }
    }

    fn ge(&self, other: &f16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            let neg = self.0 & 0x8000u16 != 0;
            let other_neg = other.0 & 0x8000u16 != 0;
            match (neg, other_neg) {
                (false, false) => self.0 >= other.0,
                (false, true) => true,
                (true, false) => (self.0 | other.0) & 0x7FFFu16 == 0,
                (true, true) => self.0 <= other.0,
            }
        }
    }
}

impl FromStr for f16 {
    type Err = ParseFloatError;
    fn from_str(src: &str) -> Result<f16, ParseFloatError> {
        f32::from_str(src).map(|x| f16::from_f32(x))
    }
}

impl Debug for f16 {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "0x{:X}", self.0)
    }
}

impl Display for f16 {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{}", self.to_f32())
    }
}

impl LowerExp for f16 {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:e}", self.to_f32())
    }
}

impl UpperExp for f16 {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:E}", self.to_f32())
    }
}

#[cfg(any(test, not(feature = "use-intrinsics")))]
mod f32_f64_bits {
    use core::mem;

    #[inline]
    pub fn f32_from_bits(bits: u32) -> f32 {
        unsafe { mem::transmute(bits) }
    }

    #[inline]
    pub fn f32_to_bits(val: f32) -> u32 {
        unsafe { mem::transmute(val) }
    }

    #[inline]
    pub fn f64_from_bits(bits: u64) -> f64 {
        unsafe { mem::transmute(bits) }
    }

    #[inline]
    pub fn f64_to_bits(val: f64) -> u64 {
        unsafe { mem::transmute(val) }
    }
}

#[cfg(feature = "use-intrinsics")]
mod convert {
    extern "C" {
        #[link_name = "llvm.convert.to.fp16.f32"]
        fn convert_to_fp16_f32(f: f32) -> u16;

        #[link_name = "llvm.convert.to.fp16.f64"]
        fn convert_to_fp16_f64(f: f64) -> u16;

        #[link_name = "llvm.convert.from.fp16.f32"]
        fn convert_from_fp16_f32(i: u16) -> f32;

        #[link_name = "llvm.convert.from.fp16.f64"]
        fn convert_from_fp16_f64(i: u16) -> f64;
    }

    #[inline(always)]
    pub fn f32_to_f16(f: f32) -> u16 {
        unsafe { convert_to_fp16_f32(f) }
    }

    #[inline(always)]
    pub fn f64_to_f16(f: f64) -> u16 {
        unsafe { convert_to_fp16_f64(f) }
    }

    #[inline(always)]
    pub fn f16_to_f32(i: u16) -> f32 {
        unsafe { convert_from_fp16_f32(i) }
    }

    #[inline(always)]
    pub fn f16_to_f64(i: u16) -> f64 {
        unsafe { convert_from_fp16_f64(i) }
    }
}

#[cfg(not(feature = "use-intrinsics"))]
mod convert {
    use core;
    use super::f32_f64_bits::*;

    pub fn f32_to_f16(value: f32) -> u16 {
        // Convert to raw bytes
        let x = f32_to_bits(value);

        // Check for signed zero
        if x & 0x7FFFFFFFu32 == 0 {
            return (x >> 16) as u16;
        }

        // Extract IEEE754 components
        let sign = x & 0x80000000u32;
        let exp = x & 0x7F800000u32;
        let man = x & 0x007FFFFFu32;

        // Subnormals will underflow, so return signed zero
        if exp == 0 {
            return (sign >> 16) as u16;
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7F800000u32 {
            // A mantissa of zero is a signed Infinity
            if man == 0 {
                return ((sign >> 16) | 0x7C00u32) as u16;
            }
            // Otherwise, this is NaN
            return ((sign >> 16) | 0x7E00u32) as u16;
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = ((exp >> 23) as i32) - 127;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return (half_sign | 0x7C00u32) as u16;
        }

        // Check for underflow
        if half_exp <= 0 {
            // Check mantissa for what we can do
            if 14 - half_exp > 24 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return half_sign as u16;
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x00800000u32;
            let mut half_man = man >> (14 - half_exp);
            // Check for rounding
            if (man >> (13 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return (half_sign | half_man) as u16;
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 13;
        // Check for rounding
        if man & 0x00001000u32 != 0 {
            // Round it
            ((half_sign | half_exp | half_man) + 1) as u16
        } else {
            (half_sign | half_exp | half_man) as u16
        }
    }

    pub fn f64_to_f16(value: f64) -> u16 {
        // Convert to raw bytes, truncating the last 32-bits of mantissa; that precision will always
        // be lost on half-precision.
        let val = f64_to_bits(value);
        let x = (val >> 32) as u32;

        // Check for signed zero
        if x & 0x7FFFFFFFu32 == 0 {
            return (x >> 16) as u16;
        }

        // Extract IEEE754 components
        let sign = x & 0x80000000u32;
        let exp = x & 0x7FF00000u32;
        let man = x & 0x000FFFFFu32;

        // Subnormals will underflow, so return signed zero
        if exp == 0 {
            return (sign >> 16) as u16;
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7FF00000u32 {
            // A mantissa of zero is a signed Infinity. We also have to check the last 32 bits.
            if (man == 0) && (val as u32 == 0) {
                return ((sign >> 16) | 0x7C00u32) as u16;
            }
            // Otherwise, this is NaN
            return ((sign >> 16) | 0x7E00u32) as u16;
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = ((exp >> 20) as i64) - 1023;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return (half_sign | 0x7C00u32) as u16;
        }

        // Check for underflow
        if half_exp <= 0 {
            // Check mantissa for what we can do
            if 10 - half_exp > 21 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return half_sign as u16;
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x00100000u32;
            let mut half_man = man >> (11 - half_exp);
            // Check for rounding
            if (man >> (10 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return (half_sign | half_man) as u16;
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 10;
        // Check for rounding
        if man & 0x00000200u32 != 0 {
            // Round it
            ((half_sign | half_exp | half_man) + 1) as u16
        } else {
            (half_sign | half_exp | half_man) as u16
        }
    }

    pub fn f16_to_f32(i: u16) -> f32 {
        // Check for signed zero
        if i & 0x7FFFu16 == 0 {
            return f32_from_bits((i as u32) << 16);
        }

        let half_sign = (i & 0x8000u16) as u32;
        let half_exp = (i & 0x7C00u16) as u32;
        let half_man = (i & 0x03FFu16) as u32;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u32 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return f32_from_bits((half_sign << 16) | 0x7F800000u32);
            } else {
                // NaN, only 1st mantissa bit is set
                return core::f32::NAN;
            }
        }

        // Calculate single-precision components with adjusted exponent
        let sign = half_sign << 16;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i32) >> 10) - 15;

        // Check for subnormals, which will be normalized by adjusting exponent
        if half_exp == 0 {
            // Calculate how much to adjust the exponent by
            let e = (half_man as u16).leading_zeros() - 6;

            // Rebias and adjust exponent
            let exp = (127 - 15 - e) << 23;
            let man = (half_man << (14 + e)) & 0x7F_FF_FFu32;
            return f32_from_bits(sign | exp | man);
        }

        // Rebias exponent for a normalized normal
        let exp = ((unbiased_exp + 127) as u32) << 23;
        let man = (half_man & 0x03FFu32) << 13;
        f32_from_bits(sign | exp | man)
    }

    pub fn f16_to_f64(i: u16) -> f64 {
        // Check for signed zero
        if i & 0x7FFFu16 == 0 {
            return f64_from_bits((i as u64) << 48);
        }

        let half_sign = (i & 0x8000u16) as u64;
        let half_exp = (i & 0x7C00u16) as u64;
        let half_man = (i & 0x03FFu16) as u64;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u64 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return f64_from_bits((half_sign << 48) | 0x7FF0000000000000u64);
            } else {
                // NaN, only 1st mantissa bit is set
                return core::f64::NAN;
            }
        }

        // Calculate double-precision components with adjusted exponent
        let sign = half_sign << 48;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i64) >> 10) - 15;

        // Check for subnormals, which will be normalized by adjusting exponent
        if half_exp == 0 {
            // Calculate how much to adjust the exponent by
            let e = (half_man as u16).leading_zeros() - 6;

            // Rebias and adjust exponent
            let exp = ((1023 - 15 - e) as u64) << 52;
            let man = (half_man << (43 + e)) & 0xF_FFFF_FFFF_FFFFu64;
            return f64_from_bits(sign | exp | man);
        }

        // Rebias exponent for a normalized normal
        let exp = ((unbiased_exp + 1023) as u64) << 52;
        let man = (half_man & 0x03FFu64) << 42;
        f64_from_bits(sign | exp | man)
    }
}

/// Contains utility functions to convert between slices of `u16` bits and `f16` numbers.
pub mod slice {
    use super::f16;
    use core::slice;

    /// Reinterpret a mutable slice of `u16` bits as a mutable slice of `f16` numbers.
    // The transmuted slice has the same life time as the original,
    // Which prevents mutating the borrowed `mut [u16]` argument
    // As long as the returned `mut [f16]` is borrowed.
    #[inline]
    pub fn from_bits_mut(bits: &mut [u16]) -> &mut [f16] {
        let pointer = bits.as_ptr() as *mut f16;
        let length = bits.len();
        unsafe { slice::from_raw_parts_mut(pointer, length) }
    }

    /// Reinterpret a mutable slice of `f16` numbers as a mutable slice of `u16` bits.
    // The transmuted slice has the same life time as the original,
    // Which prevents mutating the borrowed `mut [f16]` argument
    // As long as the returned `mut [u16]` is borrowed.
    #[inline]
    pub fn to_bits_mut(bits: &mut [f16]) -> &mut [u16] {
        let pointer = bits.as_ptr() as *mut u16;
        let length = bits.len();
        unsafe { slice::from_raw_parts_mut(pointer, length) }
    }

    /// Reinterpret a slice of `u16` bits as a slice of `f16` numbers.
    // The transmuted slice has the same life time as the original
    #[inline]
    pub fn from_bits(bits: &[u16]) -> &[f16] {
        let pointer = bits.as_ptr() as *const f16;
        let length = bits.len();
        unsafe { slice::from_raw_parts(pointer, length) }
    }

    /// Reinterpret a slice of `f16` numbers as a slice of `u16` bits.
    // The transmuted slice has the same life time as the original
    #[inline]
    pub fn to_bits(bits: &[f16]) -> &[u16] {
        let pointer = bits.as_ptr() as *const u16;
        let length = bits.len();
        unsafe { slice::from_raw_parts(pointer, length) }
    }
}

/// Contains utility functions to convert between vectors of `u16` bits and `f16` vectors.
///
/// This module is only available with the `std` feature.
#[cfg(feature = "std")]
pub mod vec {
    use super::f16;
    use core::mem;

    /// Converts a vector of `u16` elements into a vector of `f16` elements.
    /// This function merely reinterprets the contents of the vector,
    /// so it's a zero-copy operation.
    #[inline]
    pub fn from_bits(bits: Vec<u16>) -> Vec<f16> {
        let mut bits = bits;

        // An f16 array has same length and capacity as u16 array
        let length = bits.len();
        let capacity = bits.capacity();

        // Actually reinterpret the contents of the Vec<u16> as f16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `f16(u16)`
        let pointer = bits.as_mut_ptr() as *mut f16;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(bits);

        // Finally construct a new Vec<f16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }

    /// Converts a vector of `f16` elements into a vector of `u16` elements.
    /// This function merely reinterprets the contents of the vector,
    /// so it's a zero-copy operation.
    #[inline]
    pub fn to_bits(numbers: Vec<f16>) -> Vec<u16> {
        let mut numbers = numbers;

        // An f16 array has same length and capacity as u16 array
        let length = numbers.len();
        let capacity = numbers.capacity();

        // Actually reinterpret the contents of the Vec<f16> as u16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `f16(u16)`
        let pointer = numbers.as_mut_ptr() as *mut u16;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(numbers);

        // Finally construct a new Vec<f16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::f32_f64_bits::*;
    use core;
    use core::cmp::Ordering;

    #[test]
    fn test_f16_consts() {
        // DIGITS
        let digits = ((consts::MANTISSA_DIGITS as f32 - 1.0) * 2f32.log10()).floor() as u32;
        assert_eq!(consts::DIGITS, digits);
        // sanity check to show test is good
        let digits32 = ((core::f32::MANTISSA_DIGITS as f32 - 1.0) * 2f32.log10()).floor() as u32;
        assert_eq!(core::f32::DIGITS, digits32);

        // EPSILON
        let one = f16::from_f32(1.0);
        let one_plus_epsilon = f16::from_bits(one.to_bits() + 1);
        let epsilon = f16::from_f32(one_plus_epsilon.to_f32() - 1.0);
        assert_eq!(consts::EPSILON, epsilon);
        // sanity check to show test is good
        let one_plus_epsilon32= f32_from_bits(f32_to_bits(1.0) + 1);
        let epsilon32 = one_plus_epsilon32 - 1f32;
        assert_eq!(core::f32::EPSILON, epsilon32);

        // MAX, MIN and MIN_POSITIVE
        let max = f16::from_bits(consts::INFINITY.to_bits() - 1);
        let min = f16::from_bits(consts::NEG_INFINITY.to_bits() - 1);
        let min_pos = f16::from_f32(2f32.powi(consts::MIN_EXP - 1));
        assert_eq!(consts::MAX, max);
        assert_eq!(consts::MIN, min);
        assert_eq!(consts::MIN_POSITIVE, min_pos);
        // sanity check to show test is good
        let max32 = f32_from_bits(f32_to_bits(core::f32::INFINITY) - 1);
        let min32 = f32_from_bits(f32_to_bits(core::f32::NEG_INFINITY) - 1);
        let min_pos32 = 2f32.powi(core::f32::MIN_EXP - 1);
        assert_eq!(core::f32::MAX, max32);
        assert_eq!(core::f32::MIN, min32);
        assert_eq!(core::f32::MIN_POSITIVE, min_pos32);

        // MIN_10_EXP and MAX_10_EXP
        let ten_to_min = 10f32.powi(consts::MIN_10_EXP);
        assert!(ten_to_min / 10.0 < consts::MIN_POSITIVE.to_f32());
        assert!(ten_to_min > consts::MIN_POSITIVE.to_f32());
        let ten_to_max = 10f32.powi(consts::MAX_10_EXP);
        assert!(ten_to_max < consts::MAX.to_f32());
        assert!(ten_to_max * 10.0 > consts::MAX.to_f32());
        // sanity check to show test is good
        let ten_to_min32 = 10f64.powi(core::f32::MIN_10_EXP);
        assert!(ten_to_min32 / 10.0 < f64::from(core::f32::MIN_POSITIVE));
        assert!(ten_to_min32 > f64::from(core::f32::MIN_POSITIVE));
        let ten_to_max32 = 10f64.powi(core::f32::MAX_10_EXP);
        assert!(ten_to_max32 < f64::from(core::f32::MAX));
        assert!(ten_to_max32 * 10.0 > f64::from(core::f32::MAX));
    }

    #[test]
    fn test_f16_consts_from_f32() {
        let one = f16::from_f32(1.0);
        let zero = f16::from_f32(0.0);
        let neg_zero = f16::from_f32(-0.0);
        let inf = f16::from_f32(core::f32::INFINITY);
        let neg_inf = f16::from_f32(core::f32::NEG_INFINITY);
        let nan = f16::from_f32(core::f32::NAN);

        assert_eq!(consts::ONE, one);
        assert_eq!(consts::ZERO, zero);
        assert!(zero.is_sign_positive());
        assert_eq!(consts::NEG_ZERO, neg_zero);
        assert!(neg_zero.is_sign_negative());
        assert_eq!(consts::INFINITY, inf);
        assert_eq!(consts::NEG_INFINITY, neg_inf);
        assert!(nan.is_nan());
        assert!(consts::NAN.is_nan());

        let e = f16::from_f32(core::f32::consts::E);
        let pi = f16::from_f32(core::f32::consts::PI);
        let frac_1_pi = f16::from_f32(core::f32::consts::FRAC_1_PI);
        let frac_1_sqrt_2 = f16::from_f32(core::f32::consts::FRAC_1_SQRT_2);
        let frac_2_pi = f16::from_f32(core::f32::consts::FRAC_2_PI);
        let frac_2_sqrt_pi = f16::from_f32(core::f32::consts::FRAC_2_SQRT_PI);
        let frac_pi_2 = f16::from_f32(core::f32::consts::FRAC_PI_2);
        let frac_pi_3 = f16::from_f32(core::f32::consts::FRAC_PI_3);
        let frac_pi_4 = f16::from_f32(core::f32::consts::FRAC_PI_4);
        let frac_pi_6 = f16::from_f32(core::f32::consts::FRAC_PI_6);
        let frac_pi_8 = f16::from_f32(core::f32::consts::FRAC_PI_8);
        let ln_10 = f16::from_f32(core::f32::consts::LN_10);
        let ln_2 = f16::from_f32(core::f32::consts::LN_2);
        let log10_e = f16::from_f32(core::f32::consts::LOG10_E);
        let log2_e = f16::from_f32(core::f32::consts::LOG2_E);
        let sqrt_2 = f16::from_f32(core::f32::consts::SQRT_2);

        assert_eq!(consts::E, e);
        assert_eq!(consts::PI, pi);
        assert_eq!(consts::FRAC_1_PI, frac_1_pi);
        assert_eq!(consts::FRAC_1_SQRT_2, frac_1_sqrt_2);
        assert_eq!(consts::FRAC_2_PI, frac_2_pi);
        assert_eq!(consts::FRAC_2_SQRT_PI, frac_2_sqrt_pi);
        assert_eq!(consts::FRAC_PI_2, frac_pi_2);
        assert_eq!(consts::FRAC_PI_3, frac_pi_3);
        assert_eq!(consts::FRAC_PI_4, frac_pi_4);
        assert_eq!(consts::FRAC_PI_6, frac_pi_6);
        assert_eq!(consts::FRAC_PI_8, frac_pi_8);
        assert_eq!(consts::LN_10, ln_10);
        assert_eq!(consts::LN_2, ln_2);
        assert_eq!(consts::LOG10_E, log10_e);
        assert_eq!(consts::LOG2_E, log2_e);
        assert_eq!(consts::SQRT_2, sqrt_2);
    }

    #[test]
    fn test_f16_consts_from_f64() {
        let one = f16::from_f64(1.0);
        let zero = f16::from_f64(0.0);
        let neg_zero = f16::from_f64(-0.0);
        let inf = f16::from_f64(core::f64::INFINITY);
        let neg_inf = f16::from_f64(core::f64::NEG_INFINITY);
        let nan = f16::from_f64(core::f64::NAN);

        assert_eq!(consts::ONE, one);
        assert_eq!(consts::ZERO, zero);
        assert!(zero.is_sign_positive());
        assert_eq!(consts::NEG_ZERO, neg_zero);
        assert!(neg_zero.is_sign_negative());
        assert_eq!(consts::INFINITY, inf);
        assert_eq!(consts::NEG_INFINITY, neg_inf);
        assert!(nan.is_nan());
        assert!(consts::NAN.is_nan());

        let e = f16::from_f64(core::f64::consts::E);
        let pi = f16::from_f64(core::f64::consts::PI);
        let frac_1_pi = f16::from_f64(core::f64::consts::FRAC_1_PI);
        let frac_1_sqrt_2 = f16::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let frac_2_pi = f16::from_f64(core::f64::consts::FRAC_2_PI);
        let frac_2_sqrt_pi = f16::from_f64(core::f64::consts::FRAC_2_SQRT_PI);
        let frac_pi_2 = f16::from_f64(core::f64::consts::FRAC_PI_2);
        let frac_pi_3 = f16::from_f64(core::f64::consts::FRAC_PI_3);
        let frac_pi_4 = f16::from_f64(core::f64::consts::FRAC_PI_4);
        let frac_pi_6 = f16::from_f64(core::f64::consts::FRAC_PI_6);
        let frac_pi_8 = f16::from_f64(core::f64::consts::FRAC_PI_8);
        let ln_10 = f16::from_f64(core::f64::consts::LN_10);
        let ln_2 = f16::from_f64(core::f64::consts::LN_2);
        let log10_e = f16::from_f64(core::f64::consts::LOG10_E);
        let log2_e = f16::from_f64(core::f64::consts::LOG2_E);
        let sqrt_2 = f16::from_f64(core::f64::consts::SQRT_2);

        assert_eq!(consts::E, e);
        assert_eq!(consts::PI, pi);
        assert_eq!(consts::FRAC_1_PI, frac_1_pi);
        assert_eq!(consts::FRAC_1_SQRT_2, frac_1_sqrt_2);
        assert_eq!(consts::FRAC_2_PI, frac_2_pi);
        assert_eq!(consts::FRAC_2_SQRT_PI, frac_2_sqrt_pi);
        assert_eq!(consts::FRAC_PI_2, frac_pi_2);
        assert_eq!(consts::FRAC_PI_3, frac_pi_3);
        assert_eq!(consts::FRAC_PI_4, frac_pi_4);
        assert_eq!(consts::FRAC_PI_6, frac_pi_6);
        assert_eq!(consts::FRAC_PI_8, frac_pi_8);
        assert_eq!(consts::LN_10, ln_10);
        assert_eq!(consts::LN_2, ln_2);
        assert_eq!(consts::LOG10_E, log10_e);
        assert_eq!(consts::LOG2_E, log2_e);
        assert_eq!(consts::SQRT_2, sqrt_2);
    }

    #[test]
    fn test_nan_conversion() {
        let nan64 = f64_from_bits(0x7ff0_0000_0000_0001u64);
        let neg_nan64 = f64_from_bits(0xfff0_0000_0000_0001u64);
        let nan32 = f32_from_bits(0x7f80_0001u32);
        let neg_nan32 = f32_from_bits(0xff80_0001u32);
        let nan32_from_64 = nan64 as f32;
        let neg_nan32_from_64 = neg_nan64 as f32;
        let nan16_from_64 = f16::from_f64(nan64);
        let neg_nan16_from_64 = f16::from_f64(neg_nan64);
        let nan16_from_32 = f16::from_f32(nan32);
        let neg_nan16_from_32 = f16::from_f32(neg_nan32);

        assert!(nan64.is_nan());
        assert!(neg_nan64.is_nan());
        assert!(nan32.is_nan());
        assert!(neg_nan32.is_nan());
        assert!(nan32_from_64.is_nan());
        assert!(neg_nan32_from_64.is_nan());
        assert!(nan16_from_64.is_nan());
        assert!(neg_nan16_from_64.is_nan());
        assert!(nan16_from_32.is_nan());
        assert!(neg_nan16_from_32.is_nan());

        let sign64 = 1u64 << 63;
        let sign32 = 1u32 << 31;
        let sign16 = 1u16 << 15;
        let nan64_u = f64_to_bits(nan64);
        let neg_nan64_u = f64_to_bits(neg_nan64);
        let nan32_u = f32_to_bits(nan32);
        let neg_nan32_u = f32_to_bits(neg_nan32);
        let nan32_from_64_u = f32_to_bits(nan32_from_64);
        let neg_nan32_from_64_u = f32_to_bits(neg_nan32_from_64);
        let nan16_from_64_u = nan16_from_64.to_bits();
        let neg_nan16_from_64_u = neg_nan16_from_64.to_bits();
        let nan16_from_32_u = nan16_from_32.to_bits();
        let neg_nan16_from_32_u = neg_nan16_from_32.to_bits();
        assert_eq!(nan64_u & sign64, 0);
        assert_eq!(neg_nan64_u & sign64, sign64);
        assert_eq!(nan32_u & sign32, 0);
        assert_eq!(neg_nan32_u & sign32, sign32);
        assert_eq!(nan32_from_64_u & sign32, 0);
        assert_eq!(neg_nan32_from_64_u & sign32, sign32);
        assert_eq!(nan16_from_64_u & sign16, 0);
        assert_eq!(neg_nan16_from_64_u & sign16, sign16);
        assert_eq!(nan16_from_32_u & sign16, 0);
        assert_eq!(neg_nan16_from_32_u & sign16, sign16);
    }

    #[test]
    fn test_f16_to_f32() {
        let f = f16::from_f32(7.0);
        assert_eq!(f.to_f32(), 7.0f32);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = f16::from_f32(7.1);
        let diff = (f.to_f32() - 7.1f32).abs();
        // diff must be <= 4 * EPSILON, as 7 has two more significant bits than 1
        assert!(diff <= 4.0 * consts::EPSILON.to_f32());

        assert_eq!(f16::from_bits(0x0000_0001).to_f32(), 2.0f32.powi(-24));
        assert_eq!(f16::from_bits(0x0000_0005).to_f32(), 5.0 * 2.0f32.powi(-24));

        assert_eq!(f16::from_bits(0x0000_0001), f16::from_f32(2.0f32.powi(-24)));
        assert_eq!(
            f16::from_bits(0x0000_0005),
            f16::from_f32(5.0 * 2.0f32.powi(-24))
        );
    }

    #[test]
    fn test_f16_to_f64() {
        let f = f16::from_f64(7.0);
        assert_eq!(f.to_f64(), 7.0f64);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = f16::from_f64(7.1);
        let diff = (f.to_f64() - 7.1f64).abs();
        // diff must be <= 4 * EPSILON, as 7 has two more significant bits than 1
        assert!(diff <= 4.0 * consts::EPSILON.to_f64());

        assert_eq!(f16::from_bits(0x0000_0001).to_f64(), 2.0f64.powi(-24));
        assert_eq!(f16::from_bits(0x0000_0005).to_f64(), 5.0 * 2.0f64.powi(-24));

        assert_eq!(f16::from_bits(0x0000_0001), f16::from_f64(2.0f64.powi(-24)));
        assert_eq!(
            f16::from_bits(0x0000_0005),
            f16::from_f64(5.0 * 2.0f64.powi(-24))
        );
    }

    #[test]
    fn test_comparisons() {
        let zero = f16::from_f64(0.0);
        let one = f16::from_f64(1.0);
        let neg_zero = f16::from_f64(-0.0);
        let neg_one = f16::from_f64(-1.0);

        assert_eq!(zero.partial_cmp(&neg_zero), Some(Ordering::Equal));
        assert_eq!(neg_zero.partial_cmp(&zero), Some(Ordering::Equal));
        assert!(zero == neg_zero);
        assert!(neg_zero == zero);
        assert!(!(zero != neg_zero));
        assert!(!(neg_zero != zero));
        assert!(!(zero < neg_zero));
        assert!(!(neg_zero < zero));
        assert!(zero <= neg_zero);
        assert!(neg_zero <= zero);
        assert!(!(zero > neg_zero));
        assert!(!(neg_zero > zero));
        assert!(zero >= neg_zero);
        assert!(neg_zero >= zero);

        assert_eq!(one.partial_cmp(&neg_zero), Some(Ordering::Greater));
        assert_eq!(neg_zero.partial_cmp(&one), Some(Ordering::Less));
        assert!(!(one == neg_zero));
        assert!(!(neg_zero == one));
        assert!(one != neg_zero);
        assert!(neg_zero != one);
        assert!(!(one < neg_zero));
        assert!(neg_zero < one);
        assert!(!(one <= neg_zero));
        assert!(neg_zero <= one);
        assert!(one > neg_zero);
        assert!(!(neg_zero > one));
        assert!(one >= neg_zero);
        assert!(!(neg_zero >= one));

        assert_eq!(one.partial_cmp(&neg_one), Some(Ordering::Greater));
        assert_eq!(neg_one.partial_cmp(&one), Some(Ordering::Less));
        assert!(!(one == neg_one));
        assert!(!(neg_one == one));
        assert!(one != neg_one);
        assert!(neg_one != one);
        assert!(!(one < neg_one));
        assert!(neg_one < one);
        assert!(!(one <= neg_one));
        assert!(neg_one <= one);
        assert!(one > neg_one);
        assert!(!(neg_one > one));
        assert!(one >= neg_one);
        assert!(!(neg_one >= one));
    }

    #[test]
    fn test_slice_conversions() {
        use consts::*;
        let bits = &[
            E.to_bits(),
            PI.to_bits(),
            EPSILON.to_bits(),
            FRAC_1_SQRT_2.to_bits(),
        ];
        let numbers = &[E, PI, EPSILON, FRAC_1_SQRT_2];

        // Convert from bits to numbers
        let from_bits = slice::from_bits(bits);
        assert_slice_contents_eq(from_bits, numbers);

        // Convert from numbers back to bits
        let to_bits = slice::to_bits(from_bits);
        assert_slice_contents_eq(to_bits, bits);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_vec_conversions() {
        use consts::*;
        let numbers = vec![E, PI, EPSILON, FRAC_1_SQRT_2];
        let bits = vec![
            E.to_bits(),
            PI.to_bits(),
            EPSILON.to_bits(),
            FRAC_1_SQRT_2.to_bits(),
        ];
        let bits_cloned = bits.clone();

        // Convert from bits to numbers
        let from_bits = vec::from_bits(bits);
        assert_slice_contents_eq(&from_bits, &numbers);

        // Convert from numbers back to bits
        let to_bits = vec::to_bits(from_bits);
        assert_slice_contents_eq(&to_bits, &bits_cloned);
    }

    fn assert_slice_contents_eq<T: PartialEq + core::fmt::Debug>(a: &[T], b: &[T]) {
        // Checks only pointer and len,
        // but we know these are the same
        // because we just transmuted them, so
        assert_eq!(a, b);

        // We need to perform manual content equality checks
        for (a, b) in a.iter().zip(b.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_mutablility(){
        use consts::*;
        let mut bits_array = [ PI.to_bits() ];
        let bits = &mut bits_array[..];

        {   // would not compile without these braces
            // TODO: add automated test to check that it does not compile without braces
            let numbers = slice::from_bits_mut(bits);
            numbers[0] = E;
        }

        assert_eq!(bits, &[ E.to_bits() ]);

        bits[0] = LN_2.to_bits();
        assert_eq!(bits, &[ LN_2.to_bits() ]);
    }
}

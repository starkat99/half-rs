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
//! Support for `serde` crate `Serialize` and `Deserialize` traits is provided when the `serialize`
//! feature is enabled. This adds a dependency on `serde` crate so is an optional feature that works
//! on Rust 1.15 or newer.
//!
//! The crate uses `#[no_std]` by default, so can be used in embedded environments without using the
//! Rust `std` library. While a `std` feature is available, at present there are no additional
//! changes when the feature is enabled and is merely provided for forward-compatibility.

#![warn(missing_docs,
        missing_copy_implementations,
        missing_debug_implementations,
        trivial_casts,
        trivial_numeric_casts,
        unused_extern_crates,
        unused_import_braces,
        unused_qualifications)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "use-intrinsics", feature(link_llvm_intrinsics))]

#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde_derive;

#[cfg(feature = "std")]
extern crate core;

use core::num::{FpCategory, ParseFloatError};
use core::cmp::Ordering;
use core::str::FromStr;
use core::fmt::{Debug, Display, LowerExp, UpperExp, Formatter, Error};

/// The 16-bit floating point type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct f16(u16);

pub mod consts {
    //! Useful `f16` constants.

    use super::f16;

    /// 16-bit equivalent of `std::f32::DIGITS`
    pub const DIGITS: u32 = 3;
    /// 16-bit floating point epsilon. `9.7656e-4`
    pub const EPSILON: f16 = f16(0x1700u16);
    /// 16-bit positive infinity.
    pub const INFINITY: f16 = f16(0x7C00u16);
    /// 16-bit equivalent of `std::f32::MANTISSA_DIGITS`
    pub const MANTISSA_DIGITS: u32 = 11;
    /// Largest finite `f16` value. `65504`
    pub const MAX: f16 = f16(0x7BFF);
    /// 16-bit equivalent of `std::f32::MAX_10_EXP`
    pub const MAX_10_EXP: i32 = 9;
    /// 16-bit equivalent of `std::f32::MAX_EXP`
    pub const MAX_EXP: i32 = 15;
    /// Smallest finite `f16` value.
    pub const MIN: f16 = f16(0xFBFF);
    /// 16-bit equivalent of `std::f32::MIN_10_EXP`
    pub const MIN_10_EXP: i32 = -9;
    /// 16-bit equivalent of `std::f32::MIN_EXP`
    pub const MIN_EXP: i32 = -14;
    /// Smallest positive, normalized `f16` value. Approx. `6.10352e−5`
    pub const MIN_POSITIVE: f16 = f16(0x0400u16);
    /// 16-bit NaN.
    pub const NAN: f16 = f16(0xFE00u16);
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
    pub fn as_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f16` value in a `f32` value.
    ///
    /// This conversion is lossless as all 16-bit floating point values can be represented exactly
    /// in 32-bit floating point.
    #[inline]
    fn to_f32(self) -> f32 {
        convert::f16_to_f32(self.0)
    }

    /// Converts an `f16` value in a `f64` value.
    ///
    /// This conversion is lossless as all 16-bit floating point values can be represented exactly
    /// in 64-bit floating point.
    #[inline]
    fn to_f64(self) -> f64 {
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
        (self.0 & 0x7C00u16 == 0x7C00u16) && (self.0 & 0x03FFu16 != 0)
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
        (self.0 & 0x7C00u16 == 0x7C00u16) && (self.0 & 0x03FFu16 == 0)
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

    /// Returns `true` if `self`'s sign bit is positive, including `+0.0` and `INFINITY`.
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
    /// // Requires both tests to determine if is `NaN`
    /// assert!(!nan.is_sign_positive() && !nan.is_sign_negative());
    /// ```
    #[inline]
    pub fn is_sign_positive(self) -> bool {
        !self.is_nan() && self.0 & 0x8000u16 == 0
    }

    /// Returns `true` if self's sign is negative, including `-0.0` and `NEG_INFINITY`.
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
    /// // Requires both tests to determine if is `NaN`.
    /// assert!(!nan.is_sign_positive() && !nan.is_sign_negative());
    /// ```
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        !self.is_nan() && self.0 & 0x8000u16 != 0
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
        !self.is_nan() && !other.is_nan() && self.0 == other.0
    }
}

impl PartialOrd for f16 {
    fn partial_cmp(&self, other: &f16) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else if self.0 == other.0 {
            Some(Ordering::Equal)
        } else if self.0 < other.0 {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }

    fn lt(&self, other: &f16) -> bool {
        !self.is_nan() && !other.is_nan() && self.0 < other.0
    }

    fn le(&self, other: &f16) -> bool {
        !self.is_nan() && !other.is_nan() && self.0 <= other.0
    }

    fn gt(&self, other: &f16) -> bool {
        !self.is_nan() && !other.is_nan() && self.0 > other.0
    }

    fn ge(&self, other: &f16) -> bool {
        !self.is_nan() && !other.is_nan() && self.0 >= other.0
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
    use core::mem;

    use super::*;

    pub fn f32_to_f16(value: f32) -> u16 {
        // Convert to raw bytes
        let x: u32 = unsafe { mem::transmute(value) };

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
            return consts::NAN.0;
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
        let val: u64 = unsafe { mem::transmute(value) };
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
            // A mantissa of zero is a signed Infinity
            if man == 0 {
                return ((sign >> 16) | 0x7C00u32) as u16;
            }
            // Otherwise, this is NaN
            return consts::NAN.0;
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
            return unsafe { mem::transmute((i as u32) << 16) };
        }

        let half_sign = (i & 0x8000u16) as u32;
        let half_exp = (i & 0x7C00u16) as u32;
        let half_man = (i & 0x03FFu16) as u32;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u32 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return unsafe { mem::transmute((half_sign << 16) | 0x7F800000u32) };
            } else {
                // NaN, only 1st mantissa bit is set
                return core::f32::NAN;
            }
        }

        // Calculate single-precision components with adjusted exponent
        let sign = half_sign << 16;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i32) >> 10) - 15;
        let man = (half_man & 0x03FFu32) << 13;

        // Check for subnormals, which will be normalized by adjusting exponent
        if half_exp == 0 {
            // Calculate how much to adjust the exponent by
            let e = {
                let mut e_adj = 0;
                let mut hm_adj = half_man << 1;
                while hm_adj & 0x0400u32 == 0 {
                    e_adj += 1;
                    hm_adj <<= 1;
                }
                e_adj
            };

            // Rebias and adjust exponent
            let exp = ((unbiased_exp + 127 - e) << 23) as u32;
            return unsafe { mem::transmute(sign | exp | man) };
        }

        // Rebias exponent for a normalized normal
        let exp = ((unbiased_exp + 127) << 23) as u32;
        unsafe { mem::transmute(sign | exp | man) }
    }

    pub fn f16_to_f64(i: u16) -> f64 {
        // Check for signed zero
        if i & 0x7FFFu16 == 0 {
            return unsafe { mem::transmute((i as u64) << 48) };
        }

        let half_sign = (i & 0x8000u16) as u64;
        let half_exp = (i & 0x7C00u16) as u64;
        let half_man = (i & 0x03FFu16) as u64;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u64 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return unsafe { mem::transmute((half_sign << 48) | 0x7FF0000000000000u64) };
            } else {
                // NaN, only 1st mantissa bit is set
                return core::f64::NAN;
            }
        }

        // Calculate double-precision components with adjusted exponent
        let sign = half_sign << 48;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i64) >> 10) - 15;
        let man = (half_man & 0x03FFu64) << 42;

        // Check for subnormals, which will be normalized by adjusting exponent
        if half_exp == 0 {
            // Calculate how much to adjust the exponent by
            let e = {
                let mut e_adj = 0;
                let mut hm_adj = half_man << 1;
                while hm_adj & 0x0400u64 == 0 {
                    e_adj += 1;
                    hm_adj <<= 1;
                }
                e_adj
            };

            // Rebias and adjust exponent
            let exp = ((unbiased_exp + 1023 - e) << 52) as u64;
            return unsafe { mem::transmute(sign | exp | man) };
        }

        // Rebias exponent for a normalized normal
        let exp = ((unbiased_exp + 1023) << 52) as u64;
        unsafe { mem::transmute(sign | exp | man) }
    }
}

#[cfg(test)]
mod test {
    use core;
    use super::*;

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
        assert_eq!(consts::NEG_ZERO, neg_zero);
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
        assert_eq!(consts::NEG_ZERO, neg_zero);
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
    fn test_f16_to_f32() {
        let f = f16::from_f32(7.0);
        assert_eq!(f.to_f32(), 7.0f32);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = f16::from_f32(7.1);
        let diff = (f.to_f32() - 7.1f32).abs();
        assert!(diff <= consts::EPSILON.to_f32());
    }

    #[test]
    fn test_f16_to_f64() {
        let f = f16::from_f64(7.0);
        assert_eq!(f.to_f64(), 7.0f64);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = f16::from_f64(7.1);
        let diff = (f.to_f64() - 7.1f64).abs();
        assert!(diff <= consts::EPSILON.to_f64());
    }
}

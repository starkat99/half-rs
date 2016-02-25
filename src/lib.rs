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
//! The ARM architecture and some GPUs provide hardware support for the type. Currently, this type
//! does not use hardware implementations.

#![warn(missing_docs,
        missing_copy_implementations,
        missing_debug_implementations,
        trivial_casts,
        trivial_numeric_casts,
        unstable_features,
        unused_extern_crates,
        unused_import_braces,
        unused_qualifications)]


use std::mem;
use std::num::{FpCategory, ParseFloatError};
use std::cmp::Ordering;
use std::str::FromStr;
use std::fmt::{Debug, Display, LowerExp, UpperExp, Formatter, Error};

/// The 16-bit floating point type.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
pub struct f16(u16);

/// 16-bit equivalent of `std::f32::DIGITS`
pub const DIGITS: u32 = 3;
/// 16-bit floating point epsilon.
pub const EPSILON: f16 = f16(0x1700u16); // 0.00097656;
/// 16-bit positive infinity.
pub const INFINITY: f16 = f16(0x7C00u16);
/// 16-bit equivalent of `std::f32::MANTISSA_DIGITS`
pub const MANTISSA_DIGITS: u32 = 11;
/// Largest finite `f16` value.
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
/// Smallest positive, normalized `f16` value.
pub const MIN_POSITIVE: f16 = f16(0x0400u16);
/// 16-bit NaN.
pub const NAN: f16 = f16(0xFE00u16);
/// 16-bit negative infinity.
pub const NEG_INFINITY: f16 = f16(0xFC00u16);
/// 16-bit equivalent of `std::f32::RADIX`
pub const RADIX: u32 = 2;

impl f16 {
    /// Constructs a 16-bit floating point value from the raw bits.
    #[inline(always)]
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
    pub fn from_f32(value: f32) -> f16 {
        // Convert to raw bytes
        let x: u32 = unsafe { mem::transmute(value) };

        // Check for signed zero
        if x & 0x7FFFFFFFu32 == 0 {
            return f16((x >> 16) as u16);
        }

        // Extract IEEE754 components
        let sign = x & 0x80000000u32;
        let exp = x & 0x7F800000u32;
        let man = x & 0x007FFFFFu32;

        // Subnormals will underflow, so return signed zero
        if exp == 0 {
            return f16((sign >> 16) as u16);
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7F800000u32 {
            // A mantissa of zero is a signed Infinity
            if man == 0 {
                return f16(((x >> 16) | 0x7C00u32) as u16);
            }
            // Otherwise, this is NaN
            return NAN;
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = ((exp >> 23) as i32) - 127;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return f16((half_sign | 0x7C00u32) as u16);
        }

        // Check for underflow
        if half_exp <= 0 {
            // Check mantissa for what we can do
            if 14 - half_exp > 24 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return f16(half_sign as u16);
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x00800000u32;
            let mut half_man = man >> (14 - half_exp);
            // Check for rounding
            if (man >> (13 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return f16((half_sign | half_man) as u16);
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 13;
        // Check for rounding
        if man & 0x00001000u32 != 0 {
            // Round it
            f16(((half_sign | half_exp | half_man) + 1) as u16)
        } else {
            f16((half_sign | half_exp | half_man) as u16)
        }
    }

    /// Constructs a 16-bit floating point value from a 64-bit floating point value.
    ///
    /// If the 64-bit value is to large to fit in 16-bits, +/- infinity will result. NaN values are
    /// preserved. 64-bit subnormal values are too tiny to be represented in 16-bits and result in
    /// +/- 0. Exponents that underflow the minimum 16-bit exponent will result in 16-bit subnormals
    /// or +/- 0. All other values are truncated and rounded to the nearest representable 16-bit
    /// value.
    pub fn from_f64(value: f64) -> f16 {
        // Convert to raw bytes, truncating the last 32-bits of mantissa; that precision will always
        // be lost on half-precision.
        let val: u64 = unsafe { mem::transmute(value) };
        let x = (val >> 32) as u32;

        // Check for signed zero
        if x & 0x7FFFFFFFu32 == 0 {
            return f16((x >> 16) as u16);
        }

        // Extract IEEE754 components
        let sign = x & 0x80000000u32;
        let exp = x & 0x7FF00000u32;
        let man = x & 0x000FFFFFu32;

        // Subnormals will underflow, so return signed zero
        if exp == 0 {
            return f16((sign >> 16) as u16);
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7FF00000u32 {
            // A mantissa of zero is a signed Infinity
            if man == 0 {
                return f16(((x >> 16) | 0x7C00u32) as u16);
            }
            // Otherwise, this is NaN
            return NAN;
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = ((exp >> 20) as i64) - 1023;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return f16((half_sign | 0x7C00u32) as u16);
        }

        // Check for underflow
        if half_exp <= 0 {
            // Check mantissa for what we can do
            if 10 - half_exp > 21 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return f16(half_sign as u16);
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x00100000u32;
            let mut half_man = man >> (11 - half_exp);
            // Check for rounding
            if (man >> (10 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return f16((half_sign | half_man) as u16);
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 10;
        // Check for rounding
        if man & 0x00000200u32 != 0 {
            // Round it
            f16(((half_sign | half_exp | half_man) + 1) as u16)
        } else {
            f16((half_sign | half_exp | half_man) as u16)
        }
    }

    /// Converts an `f16` into the underlying bit representation.
    #[inline(always)]
    pub fn as_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f16` value in a `f32` value.
    ///
    /// This conversion is lossless as all 16-bit floating point values can be represented exactly
    /// in 32-bit floating point.
    fn to_f32(self) -> f32 {
        // Check for signed zero
        if self.0 & 0x7FFFu16 == 0 {
            return unsafe { mem::transmute((self.0 as u32) << 16) };
        }

        let half_sign = (self.0 & 0x8000u16) as u32;
        let half_exp = (self.0 & 0x7C00u16) as u32;
        let half_man = (self.0 & 0x03FFu16) as u32;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u32 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return unsafe { mem::transmute((half_sign << 16) | 0x7F800000u32) };
            } else {
                // NaN, only 1st mantissa bit is set
                return std::f32::NAN;
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

    /// Converts an `f16` value in a `f64` value.
    ///
    /// This conversion is lossless as all 16-bit floating point values can be represented exactly
    /// in 64-bit floating point.
    fn to_f64(self) -> f64 {
        // Check for signed zero
        if self.0 & 0x7FFFu16 == 0 {
            return unsafe { mem::transmute((self.0 as u64) << 48) };
        }

        let half_sign = (self.0 & 0x8000u16) as u64;
        let half_exp = (self.0 & 0x7C00u16) as u64;
        let half_man = (self.0 & 0x03FFu16) as u64;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u64 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return unsafe { mem::transmute((half_sign << 48) | 0x7FF0000000000000u64) };
            } else {
                // NaN, only 1st mantissa bit is set
                return std::f64::NAN;
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

    /// Returns `true` if this value is `NaN` and `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use f16::*;
    ///
    /// let nan = NAN;
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
    /// use f16::*;
    ///
    /// let f = f16::from_f32(7.0f32);
    /// let inf = INFINITY;
    /// let neg_inf = NEG_INFINITY;
    /// let nan = NAN;
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
    /// use f16::*;
    ///
    /// let f = f16::from_f32(7.0f32);
    /// let inf = INFINITY;
    /// let neg_inf = NEG_INFINITY;
    /// let nan = NAN;
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
    /// use f16::*;
    ///
    /// let min = MIN_POSITIVE;
    /// let max = MAX;
    /// let lower_than_min = f16::from_f32(1.0e-10_f32);
    /// let zero = f16::from_f32(0.0_f32);
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!NAN.is_normal());
    /// assert!(!INFINITY.is_normal());
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
    /// use f16::*;
    ///
    /// let num = f16::from_f32(12.4_f32);
    /// let inf = INFINITY;
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
    /// use f16::*;
    ///
    /// let f = f16::from_f32(3.5_f32);
    ///
    /// assert_eq!(f.signum(), f16::from_f32(1.0));
    /// assert_eq!(NEG_INFINITY.signum(), f16::from_f32(-1.0));
    ///
    /// assert!(NAN.signum().is_nan());
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
    /// use f16::*;
    ///
    /// let nan = NAN;
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
    /// use f16::*;
    ///
    /// let nan = NAN;
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
        write!(f, "{:x}", self.0)
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

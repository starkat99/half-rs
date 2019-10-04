//! The bfloat16 floating point format is a truncated 16-bit version of `f32`.
//!
//! `bf16` has approximately the same dynamic range as `f32` by having a lower precision than `f16`.
//! While `f16` has a precision of 11 bits, `bf16` has a precision of 8 bits.

use core::cmp::Ordering;
use core::fmt::{Debug, Display, Error, Formatter, LowerExp, UpperExp};
use core::num::{FpCategory, ParseFloatError};
use core::str::FromStr;

/// The bfloat16 floating point format is a truncated 16-bit version of `f32`.
///
/// `bf16` has approximately the same dynamic range as `f32` by having a lower precision than `f16`.
/// While `f16` has a precision of 11 bits, `bf16` has a precision of 8 bits.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct bf16(u16);

pub mod consts {
    //! Useful `bf16` constants.

    use super::bf16;

    /// bfloat16 equivalent of `std::f32::DIGITS`
    pub const DIGITS: u32 = 2;
    /// bfloat16 epsilon. `7.8125e-3`
    pub const EPSILON: bf16 = bf16(0x3C00u16);
    /// bfloat16 positive infinity.
    pub const INFINITY: bf16 = bf16(0x7F80u16);
    /// bfloat16 equivalent of `std::f32::MANTISSA_DIGITS`
    pub const MANTISSA_DIGITS: u32 = 8;
    /// Largest finite `bf16` value. `3.3895e+38`
    pub const MAX: bf16 = bf16(0x7F7F);
    /// bfloat16 equivalent of `std::f32::MAX_10_EXP`
    pub const MAX_10_EXP: i32 = 38;
    /// bfloat16 equivalent of `std::f32::MAX_EXP`
    pub const MAX_EXP: i32 = 128;
    /// Smallest finite `bf16` value. `-3.3895e+38`
    pub const MIN: bf16 = bf16(0xFF7F);
    /// bfloat16 equivalent of `std::f32::MIN_10_EXP`
    pub const MIN_10_EXP: i32 = -37;
    /// bfloat16 equivalent of `std::f32::MIN_EXP`
    pub const MIN_EXP: i32 = -125;
    /// Smallest positive, normalized `bf16` value. `1.1755e−38`
    pub const MIN_POSITIVE: bf16 = bf16(0x0080u16);
    /// bfloat16 NaN.
    pub const NAN: bf16 = bf16(0x7FC0u16);
    /// bfloat16 negative infinity.
    pub const NEG_INFINITY: bf16 = bf16(0xFF80u16);
    /// bfloat16 equivalent of `std::f32::RADIX`
    pub const RADIX: u32 = 2;

    /// bfloat16 minimum positive subnormal value. Approx. `9.1835e-41`
    pub const MIN_POSITIVE_SUBNORMAL: bf16 = bf16(0x0001u16);
    /// bfloat16 maximum subnormal value. Approx. `1.1663e-38`
    pub const MAX_SUBNORMAL: bf16 = bf16(0x007Fu16);

    /// bfloat16 floating point `1.0`
    pub const ONE: bf16 = bf16(0x3F80u16);
    /// bfloat16 floating point `0.0`
    pub const ZERO: bf16 = bf16(0x0000u16);
    /// bfloat16 floating point `-0.0`
    pub const NEG_ZERO: bf16 = bf16(0x8000u16);

    /// Euler's number.
    pub const E: bf16 = bf16(0x402Eu16);
    /// Archimedes' constant.
    pub const PI: bf16 = bf16(0x4049u16);
    /// 1.0/pi
    pub const FRAC_1_PI: bf16 = bf16(0x3EA3u16);
    /// 1.0/sqrt(2.0)
    pub const FRAC_1_SQRT_2: bf16 = bf16(0x3F35u16);
    /// 2.0/pi
    pub const FRAC_2_PI: bf16 = bf16(0x3F23u16);
    /// 2.0/sqrt(pi)
    pub const FRAC_2_SQRT_PI: bf16 = bf16(0x3F90u16);
    /// pi/2.0
    pub const FRAC_PI_2: bf16 = bf16(0x3FC9u16);
    /// pi/3.0
    pub const FRAC_PI_3: bf16 = bf16(0x3F86u16);
    /// pi/4.0
    pub const FRAC_PI_4: bf16 = bf16(0x3F49u16);
    /// pi/6.0
    pub const FRAC_PI_6: bf16 = bf16(0x3F06u16);
    /// pi/8.0
    pub const FRAC_PI_8: bf16 = bf16(0x3EC9u16);
    /// ln(10.0)
    pub const LN_10: bf16 = bf16(0x4013u16);
    /// ln(2.0)
    pub const LN_2: bf16 = bf16(0x3F31u16);
    /// log10(e)
    pub const LOG10_E: bf16 = bf16(0x3EDEu16);
    /// log2(e)
    pub const LOG2_E: bf16 = bf16(0x3FB9u16);
    /// sqrt(2)
    pub const SQRT_2: bf16 = bf16(0x3FB5u16);
}

impl bf16 {
    /// Constructs a bfloat16 value from the raw bits.
    #[inline]
    pub fn from_bits(bits: u16) -> bf16 {
        bf16(bits)
    }

    /// Constructs a bfloat16 value from a 32-bit floating point value.
    ///
    /// If the 32-bit value is too large to fit, ±∞ will result. NaN values are preserved.
    /// Subnormal values that are too tiny to be represented will result in ±0. All other values
    /// are truncated and rounded to the nearest representable value.
    #[inline]
    pub fn from_f32(value: f32) -> bf16 {
        bf16(convert::f32_to_bf16(value))
    }

    /// Constructs a bfloat16 value from a 64-bit floating point value.
    ///
    /// If the 64-bit value is to large to fit, ±∞ will result. NaN values are preserved.
    /// 64-bit subnormal values are too tiny to be represented and result in ±0. Exponents that
    /// underflow the minimum exponent will result in subnormals or ±0. All other values are
    /// truncated and rounded to the nearest representable value.
    #[inline]
    pub fn from_f64(value: f64) -> bf16 {
        bf16(convert::f64_to_bf16(value))
    }

    /// Converts a `bf16` into the underlying bit representation.
    #[inline]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts a `bf16` value into an `f32` value.
    ///
    /// This conversion is lossless as all values can be represented exactly in `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        convert::bf16_to_f32(self.0)
    }

    /// Converts a `bf16` value into an `f64` value.
    ///
    /// This conversion is lossless as all values can be represented exactly in `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 {
        convert::bf16_to_f64(self.0)
    }

    /// Returns `true` if this value is NaN and `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let nan = half::bfloat::consts::NAN;
    /// let f = bf16::from_f32(7.0_f32);
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[inline]
    pub fn is_nan(self) -> bool {
        self.0 & 0x7FFFu16 > 0x7F80u16
    }

    /// Returns `true` if this value is +∞ or −∞ and `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let f = bf16::from_f32(7.0f32);
    /// let inf = half::bfloat::consts::INFINITY;
    /// let neg_inf = half::bfloat::consts::NEG_INFINITY;
    /// let nan = half::bfloat::consts::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.0 & 0x7FFFu16 == 0x7F80u16
    }

    /// Returns `true` if this number is neither infinite nor NaN.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let f = bf16::from_f32(7.0f32);
    /// let inf = half::bfloat::consts::INFINITY;
    /// let neg_inf = half::bfloat::consts::NEG_INFINITY;
    /// let nan = half::bfloat::consts::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0 & 0x7F80u16 != 0x7F80u16
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal, or NaN.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let min = half::bfloat::consts::MIN_POSITIVE;
    /// let max = half::bfloat::consts::MAX;
    /// let lower_than_min = bf16::from_f32(1.0e-39_f32);
    /// let zero = bf16::from_f32(0.0_f32);
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!half::bfloat::consts::NAN.is_normal());
    /// assert!(!half::bfloat::consts::INFINITY.is_normal());
    /// // Values between 0 and `min` are subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    #[inline]
    pub fn is_normal(self) -> bool {
        let exp = self.0 & 0x7F80u16;
        exp != 0x7F80u16 && exp != 0
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
    /// use half::bfloat::bf16;
    ///
    /// let num = bf16::from_f32(12.4_f32);
    /// let inf = half::bfloat::consts::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    pub fn classify(self) -> FpCategory {
        let exp = self.0 & 0x7F80u16;
        let man = self.0 & 0x007Fu16;
        if exp == 0 {
            if man == 0 {
                FpCategory::Zero
            } else {
                FpCategory::Subnormal
            }
        } else if exp == 0x7F80u16 {
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
    /// * 1.0 if the number is positive, +0.0 or `INFINITY`
    /// * −1.0 if the number is negative, −0.0` or `NEG_INFINITY`
    /// * NaN if the number is NaN
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let f = bf16::from_f32(3.5_f32);
    ///
    /// assert_eq!(f.signum(), bf16::from_f32(1.0));
    /// assert_eq!(half::bfloat::consts::NEG_INFINITY.signum(), bf16::from_f32(-1.0));
    ///
    /// assert!(half::bfloat::consts::NAN.signum().is_nan());
    /// ```
    pub fn signum(self) -> bf16 {
        if self.is_nan() {
            self
        } else if self.0 & 0x8000u16 != 0 {
            bf16::from_f32(-1.0)
        } else {
            bf16::from_f32(1.0)
        }
    }

    /// Returns `true` if and only if `self` has a positive sign, including +0.0, NaNs with a
    /// positive sign bit and +∞.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let nan = half::bfloat::consts::NAN;
    /// let f = bf16::from_f32(7.0_f32);
    /// let g = bf16::from_f32(-7.0_f32);
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// // NaN can be either positive or negative
    /// assert!(nan.is_sign_positive() != nan.is_sign_negative());
    /// ```
    #[inline]
    pub fn is_sign_positive(self) -> bool {
        self.0 & 0x8000u16 == 0
    }

    /// Returns `true` if and only if `self` has a negative sign, including −0.0, NaNs with a
    /// negative sign bit and −∞.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use half::bfloat::bf16;
    ///
    /// let nan = half::bfloat::consts::NAN;
    /// let f = bf16::from_f32(7.0f32);
    /// let g = bf16::from_f32(-7.0f32);
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// // NaN can be either positive or negative
    /// assert!(nan.is_sign_positive() != nan.is_sign_negative());
    /// ```
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        self.0 & 0x8000u16 != 0
    }
}

impl From<bf16> for f32 {
    fn from(x: bf16) -> f32 {
        x.to_f32()
    }
}

impl From<bf16> for f64 {
    fn from(x: bf16) -> f64 {
        x.to_f64()
    }
}

impl From<i8> for bf16 {
    fn from(x: i8) -> bf16 {
        // Convert to f32, then to bf16
        bf16::from_f32(f32::from(x))
    }
}

impl From<u8> for bf16 {
    fn from(x: u8) -> bf16 {
        // Convert to f32, then to f16
        bf16::from_f32(f32::from(x))
    }
}

impl PartialEq for bf16 {
    fn eq(&self, other: &bf16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            (self.0 == other.0) || ((self.0 | other.0) & 0x7FFFu16 == 0)
        }
    }
}

impl PartialOrd for bf16 {
    fn partial_cmp(&self, other: &bf16) -> Option<Ordering> {
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

    fn lt(&self, other: &bf16) -> bool {
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

    fn le(&self, other: &bf16) -> bool {
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

    fn gt(&self, other: &bf16) -> bool {
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

    fn ge(&self, other: &bf16) -> bool {
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

impl FromStr for bf16 {
    type Err = ParseFloatError;
    fn from_str(src: &str) -> Result<bf16, ParseFloatError> {
        f32::from_str(src).map(bf16::from_f32)
    }
}

impl Debug for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "0x{:X}", self.0)
    }
}

impl Display for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.to_f32())
    }
}

impl LowerExp for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:e}", self.to_f32())
    }
}

impl UpperExp for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:E}", self.to_f32())
    }
}

mod convert {
    use core;

    pub fn f32_to_bf16(value: f32) -> u16 {
        // Convert to raw bytes
        let x = value.to_bits();

        // check for NaN
        if x & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
            let sign = x & 0x8000_0000u32;
            return ((sign >> 16) | 0x7FC0u32) as u16;
        }

        // round and shift
        ((x + 0x0000_8000u32) >> 16) as u16
    }

    pub fn f64_to_bf16(value: f64) -> u16 {
        // Convert to raw bytes, truncating the last 32-bits of mantissa; that precision will always
        // be lost on half-precision.
        let val = value.to_bits();
        let x = (val >> 32) as u32;

        // Check for signed zero
        if x & 0x7FFF_FFFFu32 == 0 {
            return (x >> 16) as u16;
        }

        // Extract IEEE754 components
        let sign = x & 0x8000_0000u32;
        let exp = x & 0x7FF0_0000u32;
        let man = x & 0x000F_FFFFu32;

        // Subnormals will underflow, so return signed zero
        if exp == 0 {
            return (sign >> 16) as u16;
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7FF0_0000u32 {
            // A mantissa of zero is a signed Infinity. We also have to check the last 32 bits.
            if (man == 0) && (val as u32 == 0) {
                return ((sign >> 16) | 0x7F80u32) as u16;
            }
            // Otherwise, this is NaN
            return ((sign >> 16) | 0x7FC0u32) as u16;
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for bfloat16 precision
        let unbiased_exp = ((exp >> 20) as i64) - 1023;
        let half_exp = unbiased_exp + 127;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0xFF {
            return (half_sign | 0x7F80u32) as u16;
        }

        // Check for underflow
        if half_exp <= 0 {
            // TODO: 21 is only in higher part
            // Check mantissa for what we can do
            if 7 - half_exp > 21 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return half_sign as u16;
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x0010_0000u32;
            let mut half_man = man >> (14 - half_exp);
            // Check for rounding
            if (man >> (7 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return (half_sign | half_man) as u16;
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 7;
        let half_man = man >> 13;
        // Check for rounding
        if man & 0x0000_1000u32 != 0 {
            // Round it
            ((half_sign | half_exp | half_man) + 1) as u16
        } else {
            (half_sign | half_exp | half_man) as u16
        }
    }

    pub fn bf16_to_f32(i: u16) -> f32 {
        f32::from_bits((i as u32) << 16)
    }

    pub fn bf16_to_f64(i: u16) -> f64 {
        // Check for signed zero
        if i & 0x7FFFu16 == 0 {
            return f64::from_bits((i as u64) << 48);
        }

        let half_sign = (i & 0x8000u16) as u64;
        let half_exp = (i & 0x7F80u16) as u64;
        let half_man = (i & 0x007Fu16) as u64;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7F80u64 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return f64::from_bits((half_sign << 48) | 0x7FF0_0000_0000_0000u64);
            } else {
                // NaN, only 1st mantissa bit is set
                return core::f64::NAN;
            }
        }

        // Calculate double-precision components with adjusted exponent
        let sign = half_sign << 48;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i64) >> 7) - 127;

        // Check for subnormals, which will be normalized by adjusting exponent
        if half_exp == 0 {
            // Calculate how much to adjust the exponent by
            let e = (half_man as u16).leading_zeros() - 9;

            // Rebias and adjust exponent
            let exp = ((1023 - 127 - e) as u64) << 52;
            let man = (half_man << (46 + e)) & 0xF_FFFF_FFFF_FFFFu64;
            return f64::from_bits(sign | exp | man);
        }

        // Rebias exponent for a normalized normal
        let exp = ((unbiased_exp + 1023) as u64) << 52;
        let man = (half_man & 0x007Fu64) << 45;
        f64::from_bits(sign | exp | man)
    }
}

/// Contains utility functions to convert between slices of `u16` bits and `bf16` numbers.
pub mod slice {
    use super::bf16;
    use core::slice;

    /// Reinterpret a mutable slice of `u16` bits as a mutable slice of `bf16` numbers.
    // The transmuted slice has the same life time as the original,
    // Which prevents mutating the borrowed `mut [u16]` argument
    // As long as the returned `mut [bf16]` is borrowed.
    #[inline]
    pub fn from_bits_mut(bits: &mut [u16]) -> &mut [bf16] {
        let pointer = bits.as_ptr() as *mut bf16;
        let length = bits.len();
        unsafe { slice::from_raw_parts_mut(pointer, length) }
    }

    /// Reinterpret a mutable slice of `bf16` numbers as a mutable slice of `u16` bits.
    // The transmuted slice has the same life time as the original,
    // Which prevents mutating the borrowed `mut [bf16]` argument
    // As long as the returned `mut [u16]` is borrowed.
    #[inline]
    pub fn to_bits_mut(bits: &mut [bf16]) -> &mut [u16] {
        let pointer = bits.as_ptr() as *mut u16;
        let length = bits.len();
        unsafe { slice::from_raw_parts_mut(pointer, length) }
    }

    /// Reinterpret a slice of `u16` bits as a slice of `bf16` numbers.
    // The transmuted slice has the same life time as the original
    #[inline]
    pub fn from_bits(bits: &[u16]) -> &[bf16] {
        let pointer = bits.as_ptr() as *const bf16;
        let length = bits.len();
        unsafe { slice::from_raw_parts(pointer, length) }
    }

    /// Reinterpret a slice of `bf16` numbers as a slice of `u16` bits.
    // The transmuted slice has the same life time as the original
    #[inline]
    pub fn to_bits(bits: &[bf16]) -> &[u16] {
        let pointer = bits.as_ptr() as *const u16;
        let length = bits.len();
        unsafe { slice::from_raw_parts(pointer, length) }
    }
}

/// Contains utility functions to convert between vectors of `u16` bits and `bf16` vectors.
///
/// This module is only available with the `std` feature.
#[cfg(feature = "std")]
pub mod vec {
    use super::bf16;
    use core::mem;

    /// Converts a vector of `u16` elements into a vector of `bf16` elements.
    /// This function merely reinterprets the contents of the vector,
    /// so it's a zero-copy operation.
    #[inline]
    pub fn from_bits(bits: Vec<u16>) -> Vec<bf16> {
        let mut bits = bits;

        // A bf16 array has same length and capacity as u16 array
        let length = bits.len();
        let capacity = bits.capacity();

        // Actually reinterpret the contents of the Vec<u16> as bf16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `bf16(u16)`
        let pointer = bits.as_mut_ptr() as *mut bf16;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(bits);

        // Finally construct a new Vec<bf16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }

    /// Converts a vector of `bf16` elements into a vector of `u16` elements.
    /// This function merely reinterprets the contents of the vector,
    /// so it's a zero-copy operation.
    #[inline]
    pub fn to_bits(numbers: Vec<bf16>) -> Vec<u16> {
        let mut numbers = numbers;

        // A bf16 array has same length and capacity as u16 array
        let length = numbers.len();
        let capacity = numbers.capacity();

        // Actually reinterpret the contents of the Vec<bf16> as u16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `bf16(u16)`
        let pointer = numbers.as_mut_ptr() as *mut u16;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(numbers);

        // Finally construct a new Vec<bf16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use core;
    use core::cmp::Ordering;

    #[test]
    fn test_bf16_consts_from_f32() {
        let one = bf16::from_f32(1.0);
        let zero = bf16::from_f32(0.0);
        let neg_zero = bf16::from_f32(-0.0);
        let inf = bf16::from_f32(core::f32::INFINITY);
        let neg_inf = bf16::from_f32(core::f32::NEG_INFINITY);
        let nan = bf16::from_f32(core::f32::NAN);

        assert_eq!(consts::ONE, one);
        assert_eq!(consts::ZERO, zero);
        assert_eq!(consts::NEG_ZERO, neg_zero);
        assert_eq!(consts::INFINITY, inf);
        assert_eq!(consts::NEG_INFINITY, neg_inf);
        assert!(nan.is_nan());
        assert!(consts::NAN.is_nan());

        let e = bf16::from_f32(core::f32::consts::E);
        let pi = bf16::from_f32(core::f32::consts::PI);
        let frac_1_pi = bf16::from_f32(core::f32::consts::FRAC_1_PI);
        let frac_1_sqrt_2 = bf16::from_f32(core::f32::consts::FRAC_1_SQRT_2);
        let frac_2_pi = bf16::from_f32(core::f32::consts::FRAC_2_PI);
        let frac_2_sqrt_pi = bf16::from_f32(core::f32::consts::FRAC_2_SQRT_PI);
        let frac_pi_2 = bf16::from_f32(core::f32::consts::FRAC_PI_2);
        let frac_pi_3 = bf16::from_f32(core::f32::consts::FRAC_PI_3);
        let frac_pi_4 = bf16::from_f32(core::f32::consts::FRAC_PI_4);
        let frac_pi_6 = bf16::from_f32(core::f32::consts::FRAC_PI_6);
        let frac_pi_8 = bf16::from_f32(core::f32::consts::FRAC_PI_8);
        let ln_10 = bf16::from_f32(core::f32::consts::LN_10);
        let ln_2 = bf16::from_f32(core::f32::consts::LN_2);
        let log10_e = bf16::from_f32(core::f32::consts::LOG10_E);
        let log2_e = bf16::from_f32(core::f32::consts::LOG2_E);
        let sqrt_2 = bf16::from_f32(core::f32::consts::SQRT_2);

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
    fn test_bf16_consts_from_f64() {
        let one = bf16::from_f64(1.0);
        let zero = bf16::from_f64(0.0);
        let neg_zero = bf16::from_f64(-0.0);
        let inf = bf16::from_f64(core::f64::INFINITY);
        let neg_inf = bf16::from_f64(core::f64::NEG_INFINITY);
        let nan = bf16::from_f64(core::f64::NAN);

        assert_eq!(consts::ONE, one);
        assert_eq!(consts::ZERO, zero);
        assert_eq!(consts::NEG_ZERO, neg_zero);
        assert_eq!(consts::INFINITY, inf);
        assert_eq!(consts::NEG_INFINITY, neg_inf);
        assert!(nan.is_nan());
        assert!(consts::NAN.is_nan());

        let e = bf16::from_f64(core::f64::consts::E);
        let pi = bf16::from_f64(core::f64::consts::PI);
        let frac_1_pi = bf16::from_f64(core::f64::consts::FRAC_1_PI);
        let frac_1_sqrt_2 = bf16::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let frac_2_pi = bf16::from_f64(core::f64::consts::FRAC_2_PI);
        let frac_2_sqrt_pi = bf16::from_f64(core::f64::consts::FRAC_2_SQRT_PI);
        let frac_pi_2 = bf16::from_f64(core::f64::consts::FRAC_PI_2);
        let frac_pi_3 = bf16::from_f64(core::f64::consts::FRAC_PI_3);
        let frac_pi_4 = bf16::from_f64(core::f64::consts::FRAC_PI_4);
        let frac_pi_6 = bf16::from_f64(core::f64::consts::FRAC_PI_6);
        let frac_pi_8 = bf16::from_f64(core::f64::consts::FRAC_PI_8);
        let ln_10 = bf16::from_f64(core::f64::consts::LN_10);
        let ln_2 = bf16::from_f64(core::f64::consts::LN_2);
        let log10_e = bf16::from_f64(core::f64::consts::LOG10_E);
        let log2_e = bf16::from_f64(core::f64::consts::LOG2_E);
        let sqrt_2 = bf16::from_f64(core::f64::consts::SQRT_2);

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
        let nan64 = f64::from_bits(0x7ff0_0000_0000_0001u64);
        let neg_nan64 = f64::from_bits(0xfff0_0000_0000_0001u64);
        let nan32 = f32::from_bits(0x7f80_0001u32);
        let neg_nan32 = f32::from_bits(0xff80_0001u32);
        let nan32_from_64 = nan64 as f32;
        let neg_nan32_from_64 = neg_nan64 as f32;
        let nan16_from_64 = bf16::from_f64(nan64);
        let neg_nan16_from_64 = bf16::from_f64(neg_nan64);
        let nan16_from_32 = bf16::from_f32(nan32);
        let neg_nan16_from_32 = bf16::from_f32(neg_nan32);

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
        let nan64_u = nan64.to_bits();
        let neg_nan64_u = neg_nan64.to_bits();
        let nan32_u = nan32.to_bits();
        let neg_nan32_u = neg_nan32.to_bits();
        let nan32_from_64_u = nan32_from_64.to_bits();
        let neg_nan32_from_64_u = neg_nan32_from_64.to_bits();
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
    fn test_bf16_to_f32() {
        let f = bf16::from_f32(7.0);
        assert_eq!(f.to_f32(), 7.0f32);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = bf16::from_f32(7.1);
        let diff = (f.to_f32() - 7.1f32).abs();
        // diff must be <= 4 * EPSILON, as 7 has two more significant bits than 1
        assert!(diff <= 4.0 * consts::EPSILON.to_f32());

        let tiny32 = f32::from_bits(0x0001_0000u32);
        assert_eq!(bf16::from_bits(0x0001).to_f32(), tiny32);
        assert_eq!(bf16::from_bits(0x0005).to_f32(), 5.0 * tiny32);

        assert_eq!(bf16::from_bits(0x0001), bf16::from_f32(tiny32));
        assert_eq!(bf16::from_bits(0x0005), bf16::from_f32(5.0 * tiny32));
    }

    #[test]
    fn test_bf16_to_f64() {
        let f = bf16::from_f64(7.0);
        assert_eq!(f.to_f64(), 7.0f64);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = bf16::from_f64(7.1);
        let diff = (f.to_f64() - 7.1f64).abs();
        // diff must be <= 4 * EPSILON, as 7 has two more significant bits than 1
        assert!(diff <= 4.0 * consts::EPSILON.to_f64());

        let tiny64 = 2.0f64.powi(-133);
        assert_eq!(bf16::from_bits(0x0001).to_f64(), tiny64);
        assert_eq!(bf16::from_bits(0x0005).to_f64(), 5.0 * tiny64);

        assert_eq!(bf16::from_bits(0x0001), bf16::from_f64(tiny64));
        assert_eq!(bf16::from_bits(0x0005), bf16::from_f64(5.0 * tiny64));
    }

    #[test]
    fn test_comparisons() {
        let zero = bf16::from_f64(0.0);
        let one = bf16::from_f64(1.0);
        let neg_zero = bf16::from_f64(-0.0);
        let neg_one = bf16::from_f64(-1.0);

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
        use super::consts::*;
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
    fn test_mutablility() {
        use super::consts::*;
        let mut bits_array = [PI.to_bits()];
        let bits = &mut bits_array[..];

        {
            // would not compile without these braces
            // TODO: add automated test to check that it does not compile without braces
            let numbers = slice::from_bits_mut(bits);
            numbers[0] = E;
        }

        assert_eq!(bits, &[E.to_bits()]);

        bits[0] = LN_2.to_bits();
        assert_eq!(bits, &[LN_2.to_bits()]);
    }
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};

use core::{
    cmp::Ordering,
    fmt::{
        Binary, Debug, Display, Error, Formatter, LowerExp, LowerHex, Octal, UpperExp, UpperHex,
    },
    num::{FpCategory, ParseFloatError},
    str::FromStr,
};

pub(crate) mod convert;

/// A 16-bit floating point type implementing the [`bfloat16`] format
///
/// The [`bfloat16`] floating point format is a truncated 16-bit version of the IEEE 754 standard
/// `binary32`, a.k.a [`f32`]. [`bf16`] has approximately the same dynamic range as [`f32`] by
/// having a lower precision than [`f16`][crate::f16]. While [`f16`][crate::f16] has a precision of
/// 11 bits, [`bf16`] has a precision of only 8 bits.
///
/// Like [`f16`][crate::f16], [`bf16`] does not offer arithmetic operations as it is intended for
/// compact storage rather than calculations. Operations should be performed with [`f32`] or
/// higher-precision types and converted to/from [`bf16`] as necessary.
///
/// [`bfloat16`]: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(Zeroable, Pod))]
pub struct bf16(u16);

impl bf16 {
    /// Constructs a [`bf16`] value from the raw bits
    #[inline]
    pub const fn from_bits(bits: u16) -> bf16 {
        bf16(bits)
    }

    /// Constructs a [`bf16`] value from a 32-bit floating point value
    ///
    /// If the 32-bit value is too large to fit, ±∞ will result. NaN values are preserved.
    /// Subnormal values that are too tiny to be represented will result in ±0. All other values
    /// are truncated and rounded to the nearest representable value.
    #[inline]
    pub fn from_f32(value: f32) -> bf16 {
        bf16(convert::f32_to_bf16(value))
    }

    /// Constructs a [`bf16`] value from a 64-bit floating point value
    ///
    /// If the 64-bit value is to large to fit, ±∞ will result. NaN values are preserved.
    /// 64-bit subnormal values are too tiny to be represented and result in ±0. Exponents that
    /// underflow the minimum exponent will result in subnormals or ±0. All other values are
    /// truncated and rounded to the nearest representable value.
    #[inline]
    pub fn from_f64(value: f64) -> bf16 {
        bf16(convert::f64_to_bf16(value))
    }

    /// Converts a [`bf16`] into the underlying bit representation
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Returns the memory representation of the underlying bit representation as a byte array in
    /// little-endian byte order
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let bytes = bf16::from_f32(12.5).to_le_bytes();
    /// assert_eq!(bytes, [0x48, 0x41]);
    /// ```
    #[inline]
    pub fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Returns the memory representation of the underlying bit representation as a byte array in
    /// big-endian (network) byte order
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let bytes = bf16::from_f32(12.5).to_be_bytes();
    /// assert_eq!(bytes, [0x41, 0x48]);
    /// ```
    #[inline]
    pub fn to_be_bytes(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }

    /// Returns the memory representation of the underlying bit representation as a byte array in
    /// native byte order
    ///
    /// As the target platform's native endianness is used, portable code should use
    /// [`to_be_bytes`][bf16::to_be_bytes] or [`to_le_bytes`][bf16::to_le_bytes], as appropriate,
    /// instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let bytes = bf16::from_f32(12.5).to_ne_bytes();
    /// assert_eq!(bytes, if cfg!(target_endian = "big") {
    ///     [0x41, 0x48]
    /// } else {
    ///     [0x48, 0x41]
    /// });
    /// ```
    #[inline]
    pub fn to_ne_bytes(self) -> [u8; 2] {
        self.0.to_ne_bytes()
    }

    /// Creates a floating point value from its representation as a byte array in little endian
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let value = bf16::from_le_bytes([0x48, 0x41]);
    /// assert_eq!(value, bf16::from_f32(12.5));
    /// ```
    #[inline]
    pub fn from_le_bytes(bytes: [u8; 2]) -> bf16 {
        bf16::from_bits(u16::from_le_bytes(bytes))
    }

    /// Creates a floating point value from its representation as a byte array in big endian
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let value = bf16::from_be_bytes([0x41, 0x48]);
    /// assert_eq!(value, bf16::from_f32(12.5));
    /// ```
    #[inline]
    pub fn from_be_bytes(bytes: [u8; 2]) -> bf16 {
        bf16::from_bits(u16::from_be_bytes(bytes))
    }

    /// Creates a floating point value from its representation as a byte array in native endian
    ///
    /// As the target platform's native endianness is used, portable code likely wants to use
    /// [`from_be_bytes`][bf16::from_be_bytes] or [`from_le_bytes`][bf16::from_le_bytes], as
    /// appropriate instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let value = bf16::from_ne_bytes(if cfg!(target_endian = "big") {
    ///     [0x41, 0x48]
    /// } else {
    ///     [0x48, 0x41]
    /// });
    /// assert_eq!(value, bf16::from_f32(12.5));
    /// ```
    #[inline]
    pub fn from_ne_bytes(bytes: [u8; 2]) -> bf16 {
        bf16::from_bits(u16::from_ne_bytes(bytes))
    }

    /// Converts a [`bf16`] value into an [`f32`] value
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    pub fn to_f32(self) -> f32 {
        convert::bf16_to_f32(self.0)
    }

    /// Converts a [`bf16`] value into an [`f64`] value
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    pub fn to_f64(self) -> f64 {
        convert::bf16_to_f64(self.0)
    }

    /// Returns `true` if this value is NaN and `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let nan = bf16::NAN;
    /// let f = bf16::from_f32(7.0_f32);
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[inline]
    pub const fn is_nan(self) -> bool {
        self.0 & 0x7FFFu16 > 0x7F80u16
    }

    /// Returns `true` if this value is ±∞ and `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let f = bf16::from_f32(7.0f32);
    /// let inf = bf16::INFINITY;
    /// let neg_inf = bf16::NEG_INFINITY;
    /// let nan = bf16::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.0 & 0x7FFFu16 == 0x7F80u16
    }

    /// Returns `true` if this number is neither infinite nor NaN
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let f = bf16::from_f32(7.0f32);
    /// let inf = bf16::INFINITY;
    /// let neg_inf = bf16::NEG_INFINITY;
    /// let nan = bf16::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[inline]
    pub const fn is_finite(self) -> bool {
        self.0 & 0x7F80u16 != 0x7F80u16
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal, or NaN
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let min = bf16::MIN_POSITIVE;
    /// let max = bf16::MAX;
    /// let lower_than_min = bf16::from_f32(1.0e-39_f32);
    /// let zero = bf16::from_f32(0.0_f32);
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!bf16::NAN.is_normal());
    /// assert!(!bf16::INFINITY.is_normal());
    /// // Values between 0 and `min` are subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    #[inline]
    pub fn is_normal(self) -> bool {
        let exp = self.0 & 0x7F80u16;
        exp != 0x7F80u16 && exp != 0
    }

    /// Returns the floating point category of the number
    ///
    /// If only one property is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::FpCategory;
    /// # use half::prelude::*;
    ///
    /// let num = bf16::from_f32(12.4_f32);
    /// let inf = bf16::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    pub fn classify(self) -> FpCategory {
        let exp = self.0 & 0x7F80u16;
        let man = self.0 & 0x007Fu16;
        match (exp, man) {
            (0, 0) => FpCategory::Zero,
            (0, _) => FpCategory::Subnormal,
            (0x7F80u16, 0) => FpCategory::Infinite,
            (0x7F80u16, _) => FpCategory::Nan,
            _ => FpCategory::Normal,
        }
    }

    /// Returns a number that represents the sign of `self`
    ///
    /// * 1.0 if the number is positive, +0.0 or [`INFINITY`][bf16::INFINITY]
    /// * −1.0 if the number is negative, −0.0` or [`NEG_INFINITY`][bf16::NEG_INFINITY]
    /// * [`NAN`][bf16::NAN] if the number is NaN
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let f = bf16::from_f32(3.5_f32);
    ///
    /// assert_eq!(f.signum(), bf16::from_f32(1.0));
    /// assert_eq!(bf16::NEG_INFINITY.signum(), bf16::from_f32(-1.0));
    ///
    /// assert!(bf16::NAN.signum().is_nan());
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
    /// positive sign bit and +∞
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let nan = bf16::NAN;
    /// let f = bf16::from_f32(7.0_f32);
    /// let g = bf16::from_f32(-7.0_f32);
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// // NaN can be either positive or negative
    /// assert!(nan.is_sign_positive() != nan.is_sign_negative());
    /// ```
    #[inline]
    pub const fn is_sign_positive(self) -> bool {
        self.0 & 0x8000u16 == 0
    }

    /// Returns `true` if and only if `self` has a negative sign, including −0.0, NaNs with a
    /// negative sign bit and −∞
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    ///
    /// let nan = bf16::NAN;
    /// let f = bf16::from_f32(7.0f32);
    /// let g = bf16::from_f32(-7.0f32);
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// // NaN can be either positive or negative
    /// assert!(nan.is_sign_positive() != nan.is_sign_negative());
    /// ```
    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        self.0 & 0x8000u16 != 0
    }

    /// Approximate number of [`bf16`] significant digits in base 10
    pub const DIGITS: u32 = 2;
    /// [`bf16`]
    /// [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) value
    ///
    /// This is the difference between 1.0 and the next largest representable number.
    pub const EPSILON: bf16 = bf16(0x3C00u16);
    /// [`bf16`] positive Infinity (+∞)
    pub const INFINITY: bf16 = bf16(0x7F80u16);
    /// Number of [`bf16`] significant digits in base 2
    pub const MANTISSA_DIGITS: u32 = 8;
    /// Largest finite [`bf16`] value
    pub const MAX: bf16 = bf16(0x7F7F);
    /// Maximum possible [`bf16`] power of 10 exponent
    pub const MAX_10_EXP: i32 = 38;
    /// Maximum possible [`bf16`] power of 2 exponent
    pub const MAX_EXP: i32 = 128;
    /// Smallest finite [`bf16`] value
    pub const MIN: bf16 = bf16(0xFF7F);
    /// Minimum possible normal [`bf16`] power of 10 exponent
    pub const MIN_10_EXP: i32 = -37;
    /// One greater than the minimum possible normal [`bf16`] power of 2 exponent
    pub const MIN_EXP: i32 = -125;
    /// Smallest positive normal [`bf16`] value
    pub const MIN_POSITIVE: bf16 = bf16(0x0080u16);
    /// [`bf16`] Not a Number (NaN)
    pub const NAN: bf16 = bf16(0x7FC0u16);
    /// [`bf16`] negative infinity (-∞).
    pub const NEG_INFINITY: bf16 = bf16(0xFF80u16);
    /// The radix or base of the internal representation of [`bf16`]
    pub const RADIX: u32 = 2;

    /// Minimum positive subnormal [`bf16`] value
    pub const MIN_POSITIVE_SUBNORMAL: bf16 = bf16(0x0001u16);
    /// Maximum subnormal [`bf16`] value
    pub const MAX_SUBNORMAL: bf16 = bf16(0x007Fu16);

    /// [`bf16`] 1
    pub const ONE: bf16 = bf16(0x3F80u16);
    /// [`bf16`] 0
    pub const ZERO: bf16 = bf16(0x0000u16);
    /// [`bf16`] -0
    pub const NEG_ZERO: bf16 = bf16(0x8000u16);

    /// [`bf16`] Euler's number (ℯ)
    pub const E: bf16 = bf16(0x402Eu16);
    /// [`bf16`] Archimedes' constant (π)
    pub const PI: bf16 = bf16(0x4049u16);
    /// [`bf16`] 1/π
    pub const FRAC_1_PI: bf16 = bf16(0x3EA3u16);
    /// [`bf16`] 1/√2
    pub const FRAC_1_SQRT_2: bf16 = bf16(0x3F35u16);
    /// [`bf16`] 2/π
    pub const FRAC_2_PI: bf16 = bf16(0x3F23u16);
    /// [`bf16`] 2/√π
    pub const FRAC_2_SQRT_PI: bf16 = bf16(0x3F90u16);
    /// [`bf16`] π/2
    pub const FRAC_PI_2: bf16 = bf16(0x3FC9u16);
    /// [`bf16`] π/3
    pub const FRAC_PI_3: bf16 = bf16(0x3F86u16);
    /// [`bf16`] π/4
    pub const FRAC_PI_4: bf16 = bf16(0x3F49u16);
    /// [`bf16`] π/6
    pub const FRAC_PI_6: bf16 = bf16(0x3F06u16);
    /// [`bf16`] π/8
    pub const FRAC_PI_8: bf16 = bf16(0x3EC9u16);
    /// [`bf16`] 𝗅𝗇 10
    pub const LN_10: bf16 = bf16(0x4013u16);
    /// [`bf16`] 𝗅𝗇 2
    pub const LN_2: bf16 = bf16(0x3F31u16);
    /// [`bf16`] 𝗅𝗈𝗀₁₀ℯ
    pub const LOG10_E: bf16 = bf16(0x3EDEu16);
    /// [`bf16`] 𝗅𝗈𝗀₁₀2
    pub const LOG10_2: bf16 = bf16(0x3E9Au16);
    /// [`bf16`] 𝗅𝗈𝗀₂ℯ
    pub const LOG2_E: bf16 = bf16(0x3FB9u16);
    /// [`bf16`] 𝗅𝗈𝗀₂10
    pub const LOG2_10: bf16 = bf16(0x4055u16);
    /// [`bf16`] √2
    pub const SQRT_2: bf16 = bf16(0x3FB5u16);
}

impl From<bf16> for f32 {
    #[inline]
    fn from(x: bf16) -> f32 {
        x.to_f32()
    }
}

impl From<bf16> for f64 {
    #[inline]
    fn from(x: bf16) -> f64 {
        x.to_f64()
    }
}

impl From<i8> for bf16 {
    #[inline]
    fn from(x: i8) -> bf16 {
        // Convert to f32, then to bf16
        bf16::from_f32(f32::from(x))
    }
}

impl From<u8> for bf16 {
    #[inline]
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
        write!(f, "{:?}", self.to_f32())
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

impl Binary for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:b}", self.0)
    }
}

impl Octal for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:o}", self.0)
    }
}

impl LowerHex for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:x}", self.0)
    }
}

impl UpperHex for bf16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:X}", self.0)
    }
}

#[cfg(feature = "num-traits")]
mod impl_num_traits {
    use super::bf16;
    use core::cmp::Ordering;
    use core::num::FpCategory;
    use core::ops::{Add, Mul};
    use core::ops::{Div, Neg, Rem, Sub};
    use num_traits::float::FloatCore;
    use num_traits::{
        AsPrimitive, Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive,
        Zero,
    };

    impl ToPrimitive for bf16 {
        fn to_i64(&self) -> Option<i64> {
            Self::to_f32(*self).to_i64()
        }
        fn to_u64(&self) -> Option<u64> {
            Self::to_f32(*self).to_u64()
        }
        fn to_i8(&self) -> Option<i8> {
            Self::to_f32(*self).to_i8()
        }
        fn to_u8(&self) -> Option<u8> {
            Self::to_f32(*self).to_u8()
        }
        fn to_i16(&self) -> Option<i16> {
            Self::to_f32(*self).to_i16()
        }
        fn to_u16(&self) -> Option<u16> {
            Self::to_f32(*self).to_u16()
        }
        fn to_i32(&self) -> Option<i32> {
            Self::to_f32(*self).to_i32()
        }
        fn to_u32(&self) -> Option<u32> {
            Self::to_f32(*self).to_u32()
        }
        fn to_f32(&self) -> Option<f32> {
            Some(Self::to_f32(*self))
        }
        fn to_f64(&self) -> Option<f64> {
            Some(Self::to_f64(*self))
        }
    }

    impl FromPrimitive for bf16 {
        fn from_i64(n: i64) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_u64(n: u64) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_i8(n: i8) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_u8(n: u8) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_i16(n: i16) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_u16(n: u16) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_i32(n: i32) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_u32(n: u32) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_f32(n: f32) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
        fn from_f64(n: f64) -> Option<Self> {
            n.to_f64().map(Self::from_f64)
        }
    }

    impl Num for bf16 {
        type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

        fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            Ok(Self::from_f32(f32::from_str_radix(str, radix)?))
        }
    }

    impl One for bf16 {
        #[inline]
        fn one() -> Self {
            Self::ONE
        }
    }

    impl Zero for bf16 {
        #[inline]
        fn zero() -> Self {
            Self::ZERO
        }

        #[inline]
        fn is_zero(&self) -> bool {
            *self == Self::ZERO
        }
    }

    impl Neg for bf16 {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(self.0 ^ 0x8000)
        }
    }

    impl Add for bf16 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self::from_f32(Self::to_f32(self) + Self::to_f32(rhs))
        }
    }

    impl Sub for bf16 {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self::from_f32(Self::to_f32(self) - Self::to_f32(rhs))
        }
    }

    impl Mul for bf16 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self::from_f32(Self::to_f32(self) * Self::to_f32(rhs))
        }
    }

    impl Div for bf16 {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Self::from_f32(Self::to_f32(self) / Self::to_f32(rhs))
        }
    }

    impl Rem for bf16 {
        type Output = Self;

        fn rem(self, rhs: Self) -> Self::Output {
            Self::from_f32(Self::to_f32(self) % Self::to_f32(rhs))
        }
    }

    impl NumCast for bf16 {
        fn from<T: ToPrimitive>(n: T) -> Option<Self> {
            n.to_f32().map(Self::from_f32)
        }
    }

    impl FloatCore for bf16 {
        fn infinity() -> Self {
            Self::INFINITY
        }

        fn neg_infinity() -> Self {
            Self::NEG_INFINITY
        }

        fn nan() -> Self {
            Self::NAN
        }

        fn neg_zero() -> Self {
            Self::NEG_ZERO
        }

        fn min_value() -> Self {
            Self::MIN
        }

        fn min_positive_value() -> Self {
            Self::MIN_POSITIVE
        }

        fn epsilon() -> Self {
            Self::EPSILON
        }

        fn max_value() -> Self {
            Self::MAX
        }

        fn is_nan(self) -> bool {
            self.is_nan()
        }

        fn is_infinite(self) -> bool {
            self.is_infinite()
        }

        fn is_finite(self) -> bool {
            self.is_finite()
        }

        fn is_normal(self) -> bool {
            self.is_normal()
        }

        fn classify(self) -> FpCategory {
            self.classify()
        }

        fn floor(self) -> Self {
            Self::from_f32(self.to_f32().floor())
        }

        fn ceil(self) -> Self {
            Self::from_f32(self.to_f32().ceil())
        }

        fn round(self) -> Self {
            Self::from_f32(self.to_f32().round())
        }

        fn trunc(self) -> Self {
            Self::from_f32(self.to_f32().trunc())
        }

        fn fract(self) -> Self {
            Self::from_f32(self.to_f32().fract())
        }

        fn abs(self) -> Self {
            Self(self.0 & 0x7FFF)
        }

        fn signum(self) -> Self {
            self.signum()
        }

        fn is_sign_positive(self) -> bool {
            self.is_sign_positive()
        }

        fn is_sign_negative(self) -> bool {
            self.is_sign_negative()
        }

        fn min(self, other: Self) -> Self {
            match self.partial_cmp(&other) {
                None => {
                    if self.is_nan() {
                        other
                    } else {
                        self
                    }
                }
                Some(Ordering::Greater) | Some(Ordering::Equal) => other,
                Some(Ordering::Less) => self,
            }
        }

        fn max(self, other: Self) -> Self {
            match self.partial_cmp(&other) {
                None => {
                    if self.is_nan() {
                        other
                    } else {
                        self
                    }
                }
                Some(Ordering::Greater) | Some(Ordering::Equal) => self,
                Some(Ordering::Less) => other,
            }
        }

        fn recip(self) -> Self {
            Self::from_f32(self.to_f32().recip())
        }

        fn powi(self, exp: i32) -> Self {
            Self::from_f32(self.to_f32().powi(exp))
        }

        fn to_degrees(self) -> Self {
            Self::from_f32(self.to_f32().to_degrees())
        }

        fn to_radians(self) -> Self {
            Self::from_f32(self.to_f32().to_radians())
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            FloatCore::integer_decode(self.to_f32())
        }
    }

    impl Float for bf16 {
        fn nan() -> Self {
            Self::NAN
        }

        fn infinity() -> Self {
            Self::INFINITY
        }

        fn neg_infinity() -> Self {
            Self::NEG_INFINITY
        }

        fn neg_zero() -> Self {
            Self::NEG_ZERO
        }

        fn min_value() -> Self {
            Self::MIN
        }

        fn min_positive_value() -> Self {
            Self::MIN_POSITIVE
        }

        fn epsilon() -> Self {
            Self::EPSILON
        }

        fn max_value() -> Self {
            Self::MAX
        }

        fn is_nan(self) -> bool {
            self.is_nan()
        }

        fn is_infinite(self) -> bool {
            self.is_infinite()
        }

        fn is_finite(self) -> bool {
            self.is_finite()
        }

        fn is_normal(self) -> bool {
            self.is_normal()
        }

        fn classify(self) -> FpCategory {
            self.classify()
        }

        fn floor(self) -> Self {
            Self::from_f32(self.to_f32().floor())
        }

        fn ceil(self) -> Self {
            Self::from_f32(self.to_f32().ceil())
        }

        fn round(self) -> Self {
            Self::from_f32(self.to_f32().round())
        }

        fn trunc(self) -> Self {
            Self::from_f32(self.to_f32().trunc())
        }

        fn fract(self) -> Self {
            Self::from_f32(self.to_f32().fract())
        }

        fn abs(self) -> Self {
            Self::from_f32(self.to_f32().abs())
        }

        fn signum(self) -> Self {
            Self::from_f32(self.to_f32().signum())
        }

        fn is_sign_positive(self) -> bool {
            self.is_sign_positive()
        }

        fn is_sign_negative(self) -> bool {
            self.is_sign_negative()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            Self::from_f32(self.to_f32().mul_add(a.to_f32(), b.to_f32()))
        }

        fn recip(self) -> Self {
            Self::from_f32(self.to_f32().recip())
        }

        fn powi(self, n: i32) -> Self {
            Self::from_f32(self.to_f32().powi(n))
        }

        fn powf(self, n: Self) -> Self {
            Self::from_f32(self.to_f32().powf(n.to_f32()))
        }

        fn sqrt(self) -> Self {
            Self::from_f32(self.to_f32().sqrt())
        }

        fn exp(self) -> Self {
            Self::from_f32(self.to_f32().exp())
        }

        fn exp2(self) -> Self {
            Self::from_f32(self.to_f32().exp2())
        }

        fn ln(self) -> Self {
            Self::from_f32(self.to_f32().ln())
        }

        fn log(self, base: Self) -> Self {
            Self::from_f32(self.to_f32().log(base.to_f32()))
        }

        fn log2(self) -> Self {
            Self::from_f32(self.to_f32().log2())
        }

        fn log10(self) -> Self {
            Self::from_f32(self.to_f32().log10())
        }

        fn to_degrees(self) -> Self {
            Self::from_f32(self.to_f32().to_degrees())
        }

        fn to_radians(self) -> Self {
            Self::from_f32(self.to_f32().to_radians())
        }

        fn max(self, other: Self) -> Self {
            FloatCore::max(self, other)
        }

        fn min(self, other: Self) -> Self {
            FloatCore::min(self, other)
        }

        fn abs_sub(self, other: Self) -> Self {
            Self::from_f32((self.to_f32() - other.to_f32()).max(0.0))
        }

        fn cbrt(self) -> Self {
            Self::from_f32(self.to_f32().cbrt())
        }

        fn hypot(self, other: Self) -> Self {
            Self::from_f32(self.to_f32().hypot(other.to_f32()))
        }

        fn sin(self) -> Self {
            Self::from_f32(self.to_f32().sin())
        }

        fn cos(self) -> Self {
            Self::from_f32(self.to_f32().cos())
        }

        fn tan(self) -> Self {
            Self::from_f32(self.to_f32().tan())
        }

        fn asin(self) -> Self {
            Self::from_f32(self.to_f32().asin())
        }

        fn acos(self) -> Self {
            Self::from_f32(self.to_f32().acos())
        }

        fn atan(self) -> Self {
            Self::from_f32(self.to_f32().atan())
        }

        fn atan2(self, other: Self) -> Self {
            Self::from_f32(self.to_f32().atan2(other.to_f32()))
        }

        fn sin_cos(self) -> (Self, Self) {
            let (sin, cos) = self.to_f32().sin_cos();
            (Self::from_f32(sin), Self::from_f32(cos))
        }

        fn exp_m1(self) -> Self {
            Self::from_f32(self.to_f32().exp_m1())
        }

        fn ln_1p(self) -> Self {
            Self::from_f32(self.to_f32().ln_1p())
        }

        fn sinh(self) -> Self {
            Self::from_f32(self.to_f32().sinh())
        }

        fn cosh(self) -> Self {
            Self::from_f32(self.to_f32().cosh())
        }

        fn tanh(self) -> Self {
            Self::from_f32(self.to_f32().tanh())
        }

        fn asinh(self) -> Self {
            Self::from_f32(self.to_f32().asinh())
        }

        fn acosh(self) -> Self {
            Self::from_f32(self.to_f32().acosh())
        }

        fn atanh(self) -> Self {
            Self::from_f32(self.to_f32().atanh())
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            Float::integer_decode(self.to_f32())
        }
    }

    impl FloatConst for bf16 {
        fn E() -> Self {
            Self::E
        }

        fn FRAC_1_PI() -> Self {
            Self::FRAC_1_PI
        }

        fn FRAC_1_SQRT_2() -> Self {
            Self::FRAC_1_SQRT_2
        }

        fn FRAC_2_PI() -> Self {
            Self::FRAC_2_PI
        }

        fn FRAC_2_SQRT_PI() -> Self {
            Self::FRAC_2_SQRT_PI
        }

        fn FRAC_PI_2() -> Self {
            Self::FRAC_PI_2
        }

        fn FRAC_PI_3() -> Self {
            Self::FRAC_PI_3
        }

        fn FRAC_PI_4() -> Self {
            Self::FRAC_PI_4
        }

        fn FRAC_PI_6() -> Self {
            Self::FRAC_PI_6
        }

        fn FRAC_PI_8() -> Self {
            Self::FRAC_PI_8
        }

        fn LN_10() -> Self {
            Self::LN_10
        }

        fn LN_2() -> Self {
            Self::LN_2
        }

        fn LOG10_E() -> Self {
            Self::LOG10_E
        }

        fn LOG2_E() -> Self {
            Self::LOG2_E
        }

        fn PI() -> Self {
            Self::PI
        }

        fn SQRT_2() -> Self {
            Self::SQRT_2
        }

        fn LOG10_2() -> Self
        where
            Self: Sized + Div<Self, Output = Self>,
        {
            Self::LOG10_2
        }

        fn LOG2_10() -> Self
        where
            Self: Sized + Div<Self, Output = Self>,
        {
            Self::LOG2_10
        }
    }

    impl Bounded for bf16 {
        fn min_value() -> Self {
            bf16::MIN
        }

        fn max_value() -> Self {
            bf16::MAX
        }
    }

    macro_rules! impl_as_primitive_to_bf16 {
        ($ty:ty, $meth:ident) => {
            impl AsPrimitive<$ty> for bf16 {
                fn as_(self) -> $ty {
                    self.$meth().as_()
                }
            }
        };
    }

    impl_as_primitive_to_bf16!(i64, to_f32);
    impl_as_primitive_to_bf16!(u64, to_f32);
    impl_as_primitive_to_bf16!(i8, to_f32);
    impl_as_primitive_to_bf16!(u8, to_f32);
    impl_as_primitive_to_bf16!(i16, to_f32);
    impl_as_primitive_to_bf16!(u16, to_f32);
    impl_as_primitive_to_bf16!(i32, to_f32);
    impl_as_primitive_to_bf16!(u32, to_f32);
    impl_as_primitive_to_bf16!(f32, to_f32);
    impl_as_primitive_to_bf16!(f64, to_f64);

    macro_rules! impl_as_primitive_bf16_from {
        ($ty:ty, $meth:ident) => {
            impl AsPrimitive<bf16> for $ty {
                fn as_(self) -> bf16 {
                    bf16::$meth(self.as_())
                }
            }
        };
    }

    impl_as_primitive_bf16_from!(i64, from_f32);
    impl_as_primitive_bf16_from!(u64, from_f32);
    impl_as_primitive_bf16_from!(i8, from_f32);
    impl_as_primitive_bf16_from!(u8, from_f32);
    impl_as_primitive_bf16_from!(i16, from_f32);
    impl_as_primitive_bf16_from!(u16, from_f32);
    impl_as_primitive_bf16_from!(i32, from_f32);
    impl_as_primitive_bf16_from!(u32, from_f32);
    impl_as_primitive_bf16_from!(f32, from_f32);
    impl_as_primitive_bf16_from!(f64, from_f64);
}

#[allow(
    clippy::cognitive_complexity,
    clippy::float_cmp,
    clippy::neg_cmp_op_on_partial_ord
)]
#[cfg(test)]
mod test {
    use super::*;
    use core::cmp::Ordering;
    #[cfg(feature = "num-traits")]
    use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive};
    use quickcheck_macros::quickcheck;

    #[cfg(feature = "num-traits")]
    #[test]
    fn as_primitive() {
        let two = bf16::from_f32(2.0);
        assert_eq!(<i32 as AsPrimitive<bf16>>::as_(2), two);
        assert_eq!(<bf16 as AsPrimitive<i32>>::as_(two), 2);

        assert_eq!(<f32 as AsPrimitive<bf16>>::as_(2.0), two);
        assert_eq!(<bf16 as AsPrimitive<f32>>::as_(two), 2.0);

        assert_eq!(<f64 as AsPrimitive<bf16>>::as_(2.0), two);
        assert_eq!(<bf16 as AsPrimitive<f64>>::as_(two), 2.0);
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn to_primitive() {
        let two = bf16::from_f32(2.0);
        assert_eq!(ToPrimitive::to_i32(&two).unwrap(), 2i32);
        assert_eq!(ToPrimitive::to_f32(&two).unwrap(), 2.0f32);
        assert_eq!(ToPrimitive::to_f64(&two).unwrap(), 2.0f64);
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn from_primitive() {
        let two = bf16::from_f32(2.0);
        assert_eq!(<bf16 as FromPrimitive>::from_i32(2).unwrap(), two);
        assert_eq!(<bf16 as FromPrimitive>::from_f32(2.0).unwrap(), two);
        assert_eq!(<bf16 as FromPrimitive>::from_f64(2.0).unwrap(), two);
    }

    #[test]
    fn test_bf16_consts_from_f32() {
        let one = bf16::from_f32(1.0);
        let zero = bf16::from_f32(0.0);
        let neg_zero = bf16::from_f32(-0.0);
        let inf = bf16::from_f32(core::f32::INFINITY);
        let neg_inf = bf16::from_f32(core::f32::NEG_INFINITY);
        let nan = bf16::from_f32(core::f32::NAN);

        assert_eq!(bf16::ONE, one);
        assert_eq!(bf16::ZERO, zero);
        assert_eq!(bf16::NEG_ZERO, neg_zero);
        assert_eq!(bf16::INFINITY, inf);
        assert_eq!(bf16::NEG_INFINITY, neg_inf);
        assert!(nan.is_nan());
        assert!(bf16::NAN.is_nan());

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
        // core::f32::consts::LOG10_2 requires rustc 1.43.0
        let log10_2 = bf16::from_f32(2f32.log10());
        let log2_e = bf16::from_f32(core::f32::consts::LOG2_E);
        // core::f32::consts::LOG2_10 requires rustc 1.43.0
        let log2_10 = bf16::from_f32(10f32.log2());
        let sqrt_2 = bf16::from_f32(core::f32::consts::SQRT_2);

        assert_eq!(bf16::E, e);
        assert_eq!(bf16::PI, pi);
        assert_eq!(bf16::FRAC_1_PI, frac_1_pi);
        assert_eq!(bf16::FRAC_1_SQRT_2, frac_1_sqrt_2);
        assert_eq!(bf16::FRAC_2_PI, frac_2_pi);
        assert_eq!(bf16::FRAC_2_SQRT_PI, frac_2_sqrt_pi);
        assert_eq!(bf16::FRAC_PI_2, frac_pi_2);
        assert_eq!(bf16::FRAC_PI_3, frac_pi_3);
        assert_eq!(bf16::FRAC_PI_4, frac_pi_4);
        assert_eq!(bf16::FRAC_PI_6, frac_pi_6);
        assert_eq!(bf16::FRAC_PI_8, frac_pi_8);
        assert_eq!(bf16::LN_10, ln_10);
        assert_eq!(bf16::LN_2, ln_2);
        assert_eq!(bf16::LOG10_E, log10_e);
        assert_eq!(bf16::LOG10_2, log10_2);
        assert_eq!(bf16::LOG2_E, log2_e);
        assert_eq!(bf16::LOG2_10, log2_10);
        assert_eq!(bf16::SQRT_2, sqrt_2);
    }

    #[test]
    fn test_bf16_consts_from_f64() {
        let one = bf16::from_f64(1.0);
        let zero = bf16::from_f64(0.0);
        let neg_zero = bf16::from_f64(-0.0);
        let inf = bf16::from_f64(core::f64::INFINITY);
        let neg_inf = bf16::from_f64(core::f64::NEG_INFINITY);
        let nan = bf16::from_f64(core::f64::NAN);

        assert_eq!(bf16::ONE, one);
        assert_eq!(bf16::ZERO, zero);
        assert_eq!(bf16::NEG_ZERO, neg_zero);
        assert_eq!(bf16::INFINITY, inf);
        assert_eq!(bf16::NEG_INFINITY, neg_inf);
        assert!(nan.is_nan());
        assert!(bf16::NAN.is_nan());

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
        // core::f64::consts::LOG10_2 requires rustc 1.43.0
        let log10_2 = bf16::from_f64(2f64.log10());
        let log2_e = bf16::from_f64(core::f64::consts::LOG2_E);
        // core::f64::consts::LOG2_10 requires rustc 1.43.0
        let log2_10 = bf16::from_f64(10f64.log2());
        let sqrt_2 = bf16::from_f64(core::f64::consts::SQRT_2);

        assert_eq!(bf16::E, e);
        assert_eq!(bf16::PI, pi);
        assert_eq!(bf16::FRAC_1_PI, frac_1_pi);
        assert_eq!(bf16::FRAC_1_SQRT_2, frac_1_sqrt_2);
        assert_eq!(bf16::FRAC_2_PI, frac_2_pi);
        assert_eq!(bf16::FRAC_2_SQRT_PI, frac_2_sqrt_pi);
        assert_eq!(bf16::FRAC_PI_2, frac_pi_2);
        assert_eq!(bf16::FRAC_PI_3, frac_pi_3);
        assert_eq!(bf16::FRAC_PI_4, frac_pi_4);
        assert_eq!(bf16::FRAC_PI_6, frac_pi_6);
        assert_eq!(bf16::FRAC_PI_8, frac_pi_8);
        assert_eq!(bf16::LN_10, ln_10);
        assert_eq!(bf16::LN_2, ln_2);
        assert_eq!(bf16::LOG10_E, log10_e);
        assert_eq!(bf16::LOG10_2, log10_2);
        assert_eq!(bf16::LOG2_E, log2_e);
        assert_eq!(bf16::LOG2_10, log2_10);
        assert_eq!(bf16::SQRT_2, sqrt_2);
    }

    #[test]
    fn test_nan_conversion_to_smaller() {
        let nan64 = f64::from_bits(0x7FF0_0000_0000_0001u64);
        let neg_nan64 = f64::from_bits(0xFFF0_0000_0000_0001u64);
        let nan32 = f32::from_bits(0x7F80_0001u32);
        let neg_nan32 = f32::from_bits(0xFF80_0001u32);
        let nan32_from_64 = nan64 as f32;
        let neg_nan32_from_64 = neg_nan64 as f32;
        let nan16_from_64 = bf16::from_f64(nan64);
        let neg_nan16_from_64 = bf16::from_f64(neg_nan64);
        let nan16_from_32 = bf16::from_f32(nan32);
        let neg_nan16_from_32 = bf16::from_f32(neg_nan32);

        assert!(nan64.is_nan() && nan64.is_sign_positive());
        assert!(neg_nan64.is_nan() && neg_nan64.is_sign_negative());
        assert!(nan32.is_nan() && nan32.is_sign_positive());
        assert!(neg_nan32.is_nan() && neg_nan32.is_sign_negative());
        assert!(nan32_from_64.is_nan() && nan32_from_64.is_sign_positive());
        assert!(neg_nan32_from_64.is_nan() && neg_nan32_from_64.is_sign_negative());
        assert!(nan16_from_64.is_nan() && nan16_from_64.is_sign_positive());
        assert!(neg_nan16_from_64.is_nan() && neg_nan16_from_64.is_sign_negative());
        assert!(nan16_from_32.is_nan() && nan16_from_32.is_sign_positive());
        assert!(neg_nan16_from_32.is_nan() && neg_nan16_from_32.is_sign_negative());
    }

    #[test]
    fn test_nan_conversion_to_larger() {
        let nan16 = bf16::from_bits(0x7F81u16);
        let neg_nan16 = bf16::from_bits(0xFF81u16);
        let nan32 = f32::from_bits(0x7F80_0001u32);
        let neg_nan32 = f32::from_bits(0xFF80_0001u32);
        let nan32_from_16 = f32::from(nan16);
        let neg_nan32_from_16 = f32::from(neg_nan16);
        let nan64_from_16 = f64::from(nan16);
        let neg_nan64_from_16 = f64::from(neg_nan16);
        let nan64_from_32 = f64::from(nan32);
        let neg_nan64_from_32 = f64::from(neg_nan32);

        assert!(nan16.is_nan() && nan16.is_sign_positive());
        assert!(neg_nan16.is_nan() && neg_nan16.is_sign_negative());
        assert!(nan32.is_nan() && nan32.is_sign_positive());
        assert!(neg_nan32.is_nan() && neg_nan32.is_sign_negative());
        assert!(nan32_from_16.is_nan() && nan32_from_16.is_sign_positive());
        assert!(neg_nan32_from_16.is_nan() && neg_nan32_from_16.is_sign_negative());
        assert!(nan64_from_16.is_nan() && nan64_from_16.is_sign_positive());
        assert!(neg_nan64_from_16.is_nan() && neg_nan64_from_16.is_sign_negative());
        assert!(nan64_from_32.is_nan() && nan64_from_32.is_sign_positive());
        assert!(neg_nan64_from_32.is_nan() && neg_nan64_from_32.is_sign_negative());
    }

    #[test]
    fn test_bf16_to_f32() {
        let f = bf16::from_f32(7.0);
        assert_eq!(f.to_f32(), 7.0f32);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = bf16::from_f32(7.1);
        let diff = (f.to_f32() - 7.1f32).abs();
        // diff must be <= 4 * EPSILON, as 7 has two more significant bits than 1
        assert!(diff <= 4.0 * bf16::EPSILON.to_f32());

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
        assert!(diff <= 4.0 * bf16::EPSILON.to_f64());

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
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn round_to_even_f32() {
        // smallest positive subnormal = 0b0.0000_001 * 2^-126 = 2^-133
        let min_sub = bf16::from_bits(1);
        let min_sub_f = (-133f32).exp2();
        assert_eq!(bf16::from_f32(min_sub_f).to_bits(), min_sub.to_bits());
        assert_eq!(f32::from(min_sub).to_bits(), min_sub_f.to_bits());

        // 0.0000000_011111 rounded to 0.0000000 (< tie, no rounding)
        // 0.0000000_100000 rounded to 0.0000000 (tie and even, remains at even)
        // 0.0000000_100001 rounded to 0.0000001 (> tie, rounds up)
        assert_eq!(
            bf16::from_f32(min_sub_f * 0.49).to_bits(),
            min_sub.to_bits() * 0
        );
        assert_eq!(
            bf16::from_f32(min_sub_f * 0.50).to_bits(),
            min_sub.to_bits() * 0
        );
        assert_eq!(
            bf16::from_f32(min_sub_f * 0.51).to_bits(),
            min_sub.to_bits() * 1
        );

        // 0.0000001_011111 rounded to 0.0000001 (< tie, no rounding)
        // 0.0000001_100000 rounded to 0.0000010 (tie and odd, rounds up to even)
        // 0.0000001_100001 rounded to 0.0000010 (> tie, rounds up)
        assert_eq!(
            bf16::from_f32(min_sub_f * 1.49).to_bits(),
            min_sub.to_bits() * 1
        );
        assert_eq!(
            bf16::from_f32(min_sub_f * 1.50).to_bits(),
            min_sub.to_bits() * 2
        );
        assert_eq!(
            bf16::from_f32(min_sub_f * 1.51).to_bits(),
            min_sub.to_bits() * 2
        );

        // 0.0000010_011111 rounded to 0.0000010 (< tie, no rounding)
        // 0.0000010_100000 rounded to 0.0000010 (tie and even, remains at even)
        // 0.0000010_100001 rounded to 0.0000011 (> tie, rounds up)
        assert_eq!(
            bf16::from_f32(min_sub_f * 2.49).to_bits(),
            min_sub.to_bits() * 2
        );
        assert_eq!(
            bf16::from_f32(min_sub_f * 2.50).to_bits(),
            min_sub.to_bits() * 2
        );
        assert_eq!(
            bf16::from_f32(min_sub_f * 2.51).to_bits(),
            min_sub.to_bits() * 3
        );

        assert_eq!(
            bf16::from_f32(250.49f32).to_bits(),
            bf16::from_f32(250.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(250.50f32).to_bits(),
            bf16::from_f32(250.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(250.51f32).to_bits(),
            bf16::from_f32(251.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(251.49f32).to_bits(),
            bf16::from_f32(251.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(251.50f32).to_bits(),
            bf16::from_f32(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(251.51f32).to_bits(),
            bf16::from_f32(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(252.49f32).to_bits(),
            bf16::from_f32(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(252.50f32).to_bits(),
            bf16::from_f32(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f32(252.51f32).to_bits(),
            bf16::from_f32(253.0).to_bits()
        );
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn round_to_even_f64() {
        // smallest positive subnormal = 0b0.0000_001 * 2^-126 = 2^-133
        let min_sub = bf16::from_bits(1);
        let min_sub_f = (-133f64).exp2();
        assert_eq!(bf16::from_f64(min_sub_f).to_bits(), min_sub.to_bits());
        assert_eq!(f64::from(min_sub).to_bits(), min_sub_f.to_bits());

        // 0.0000000_011111 rounded to 0.0000000 (< tie, no rounding)
        // 0.0000000_100000 rounded to 0.0000000 (tie and even, remains at even)
        // 0.0000000_100001 rounded to 0.0000001 (> tie, rounds up)
        assert_eq!(
            bf16::from_f64(min_sub_f * 0.49).to_bits(),
            min_sub.to_bits() * 0
        );
        assert_eq!(
            bf16::from_f64(min_sub_f * 0.50).to_bits(),
            min_sub.to_bits() * 0
        );
        assert_eq!(
            bf16::from_f64(min_sub_f * 0.51).to_bits(),
            min_sub.to_bits() * 1
        );

        // 0.0000001_011111 rounded to 0.0000001 (< tie, no rounding)
        // 0.0000001_100000 rounded to 0.0000010 (tie and odd, rounds up to even)
        // 0.0000001_100001 rounded to 0.0000010 (> tie, rounds up)
        assert_eq!(
            bf16::from_f64(min_sub_f * 1.49).to_bits(),
            min_sub.to_bits() * 1
        );
        assert_eq!(
            bf16::from_f64(min_sub_f * 1.50).to_bits(),
            min_sub.to_bits() * 2
        );
        assert_eq!(
            bf16::from_f64(min_sub_f * 1.51).to_bits(),
            min_sub.to_bits() * 2
        );

        // 0.0000010_011111 rounded to 0.0000010 (< tie, no rounding)
        // 0.0000010_100000 rounded to 0.0000010 (tie and even, remains at even)
        // 0.0000010_100001 rounded to 0.0000011 (> tie, rounds up)
        assert_eq!(
            bf16::from_f64(min_sub_f * 2.49).to_bits(),
            min_sub.to_bits() * 2
        );
        assert_eq!(
            bf16::from_f64(min_sub_f * 2.50).to_bits(),
            min_sub.to_bits() * 2
        );
        assert_eq!(
            bf16::from_f64(min_sub_f * 2.51).to_bits(),
            min_sub.to_bits() * 3
        );

        assert_eq!(
            bf16::from_f64(250.49f64).to_bits(),
            bf16::from_f64(250.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(250.50f64).to_bits(),
            bf16::from_f64(250.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(250.51f64).to_bits(),
            bf16::from_f64(251.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(251.49f64).to_bits(),
            bf16::from_f64(251.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(251.50f64).to_bits(),
            bf16::from_f64(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(251.51f64).to_bits(),
            bf16::from_f64(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(252.49f64).to_bits(),
            bf16::from_f64(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(252.50f64).to_bits(),
            bf16::from_f64(252.0).to_bits()
        );
        assert_eq!(
            bf16::from_f64(252.51f64).to_bits(),
            bf16::from_f64(253.0).to_bits()
        );
    }

    impl quickcheck::Arbitrary for bf16 {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            bf16(u16::arbitrary(g))
        }
    }

    #[quickcheck]
    fn qc_roundtrip_bf16_f32_is_identity(f: bf16) -> bool {
        let roundtrip = bf16::from_f32(f.to_f32());
        if f.is_nan() {
            roundtrip.is_nan() && f.is_sign_negative() == roundtrip.is_sign_negative()
        } else {
            f.0 == roundtrip.0
        }
    }

    #[quickcheck]
    fn qc_roundtrip_bf16_f64_is_identity(f: bf16) -> bool {
        let roundtrip = bf16::from_f64(f.to_f64());
        if f.is_nan() {
            roundtrip.is_nan() && f.is_sign_negative() == roundtrip.is_sign_negative()
        } else {
            f.0 == roundtrip.0
        }
    }
}

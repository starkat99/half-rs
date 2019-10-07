//! Contains utility functions to convert between vectors of `u16` bits and `f16` vectors.
//!
//! This module is only available with the `std` feature.

#![cfg(feature = "std")]

use super::{bfloat::bf16, f16};
use core::mem;

/// Extensions to `Vec<f16>` and `Vec<bf16>` to support reinterpret operations.
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait HalfFloatVecExt: private::SealedHalfFloatVec {
    /// Reinterpret a vector of `f16` or `bf16` numbers as a vector of `u16` bits.
    ///
    /// This is a zero-copy operation. The reinterpreted vector has the same memory location as
    /// `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let float_buffer = vec![f16::from_f32(1.), f16::from_f32(2.), f16::from_f32(3.)];
    /// let int_buffer = float_buffer.reinterpret_into();
    ///
    /// assert_eq!(int_buffer, [f16::from_f32(1.).to_bits(), f16::from_f32(2.).to_bits(), f16::from_f32(3.).to_bits()]);
    /// ```
    fn reinterpret_into(self) -> Vec<u16>;
}

/// Extensions to `Vec<u16>` to support reinterpret operations.
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait HalfBitsVecExt: private::SealedHalfBitsVec {
    /// Reinterpret a vector of `u16` bits as a vector of `f16` or `bf16` numbers.
    ///
    /// `H` is the type to cast to, and must be either `f16` or `bf16` type.
    ///
    /// This is a zero-copy operation. The reinterpreted vector has the same memory location as
    /// `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use half::prelude::*;
    /// let int_buffer = vec![f16::from_f32(1.).to_bits(), f16::from_f32(2.).to_bits(), f16::from_f32(3.).to_bits()];
    /// let float_buffer = int_buffer.reinterpret_into::<f16>();
    ///
    /// assert_eq!(float_buffer, [f16::from_f32(1.), f16::from_f32(2.), f16::from_f32(3.)]);
    /// ```
    fn reinterpret_into<H>(self) -> Vec<H>
    where
        H: crate::private::SealedHalf;
}

mod private {
    use crate::{bfloat::bf16, f16};

    pub trait SealedHalfFloatVec {}
    impl SealedHalfFloatVec for Vec<f16> {}
    impl SealedHalfFloatVec for Vec<bf16> {}

    pub trait SealedHalfBitsVec {}
    impl SealedHalfBitsVec for Vec<u16> {}
}

impl HalfFloatVecExt for Vec<f16> {
    fn reinterpret_into(mut self) -> Vec<u16> {
        // An f16 array has same length and capacity as u16 array
        let length = self.len();
        let capacity = self.capacity();

        // Actually reinterpret the contents of the Vec<f16> as u16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `f16(u16)`
        let pointer = self.as_mut_ptr() as *mut u16;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(self);

        // Finally construct a new Vec<f16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }
}

impl HalfFloatVecExt for Vec<bf16> {
    fn reinterpret_into(mut self) -> Vec<u16> {
        // An f16 array has same length and capacity as u16 array
        let length = self.len();
        let capacity = self.capacity();

        // Actually reinterpret the contents of the Vec<f16> as u16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `f16(u16)`
        let pointer = self.as_mut_ptr() as *mut u16;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(self);

        // Finally construct a new Vec<f16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }
}

impl HalfBitsVecExt for Vec<u16> {
    // This is safe because all traits are sealed
    #[inline]
    fn reinterpret_into<H>(mut self) -> Vec<H>
    where
        H: crate::private::SealedHalf,
    {
        // An f16 array has same length and capacity as u16 array
        let length = self.len();
        let capacity = self.capacity();

        // Actually reinterpret the contents of the Vec<u16> as f16,
        // knowing that structs are represented as only their members in memory,
        // which is the u16 part of `f16(u16)`
        let pointer = self.as_mut_ptr() as *mut H;

        // Prevent running a destructor on the old Vec<u16>, so the pointer won't be deleted
        mem::forget(self);

        // Finally construct a new Vec<f16> from the raw pointer
        unsafe { Vec::from_raw_parts(pointer, length, capacity) }
    }
}

/// Converts a vector of `u16` elements into a vector of `f16` elements.
///
/// This function merely reinterprets the contents of the vector,
/// so it's a zero-copy operation.
#[deprecated(since = "1.4.0", note = "use HalfBitsVecExt::reinterpret_into instead")]
#[inline]
pub fn from_bits(bits: Vec<u16>) -> Vec<f16> {
    bits.reinterpret_into()
}

/// Converts a vector of `f16` elements into a vector of `u16` elements.
///
/// This function merely reinterprets the contents of the vector,
/// so it's a zero-copy operation.
#[deprecated(
    since = "1.4.0",
    note = "use HalfFloatVecExt::reinterpret_into instead"
)]
#[inline]
pub fn to_bits(numbers: Vec<f16>) -> Vec<u16> {
    numbers.reinterpret_into()
}

#[cfg(test)]
mod test {
    use super::{HalfBitsVecExt, HalfFloatVecExt};
    use crate::{bfloat::bf16, f16};

    #[test]
    fn test_vec_conversions_f16() {
        use crate::consts::*;
        let numbers = vec![E, PI, EPSILON, FRAC_1_SQRT_2];
        let bits = vec![
            E.to_bits(),
            PI.to_bits(),
            EPSILON.to_bits(),
            FRAC_1_SQRT_2.to_bits(),
        ];
        let bits_cloned = bits.clone();

        // Convert from bits to numbers
        let from_bits = bits.reinterpret_into::<f16>();
        assert_eq!(&from_bits[..], &numbers[..]);

        // Convert from numbers back to bits
        let to_bits = from_bits.reinterpret_into();
        assert_eq!(&to_bits[..], &bits_cloned[..]);
    }

    #[test]
    fn test_vec_conversions_bf16() {
        use crate::bfloat::consts::*;
        let numbers = vec![E, PI, EPSILON, FRAC_1_SQRT_2];
        let bits = vec![
            E.to_bits(),
            PI.to_bits(),
            EPSILON.to_bits(),
            FRAC_1_SQRT_2.to_bits(),
        ];
        let bits_cloned = bits.clone();

        // Convert from bits to numbers
        let from_bits = bits.reinterpret_into::<bf16>();
        assert_eq!(&from_bits[..], &numbers[..]);

        // Convert from numbers back to bits
        let to_bits = from_bits.reinterpret_into();
        assert_eq!(&to_bits[..], &bits_cloned[..]);
    }
}

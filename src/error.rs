//! Error type for numeric conversion functions.

use core::fmt;

/// The error type returned when a checked integral type conversion fails.
pub struct TryFromFloatError(pub(crate) ());

impl fmt::Display for TryFromFloatError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = "out of range integral type conversion attempted";
        fmt::Display::fmt(msg, f)
    }
}

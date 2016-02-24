
use std::mem;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default)]
pub struct f16(u16);

// TODO Endianness *should* be fine -- except in archs the differ on int/fp endianness

impl f16 {
    pub fn from_bits(bits: u16) -> f16 {
        f16(bits)
    }

    pub fn from_f32(value: f32) -> f16 {
        // Convert to raw bytes
        let x: u32 = unsafe { mem::transmute(value) };

        // Check for signed zero
        if x & 0x7FFFFFFFu32 == 0 {
            return f16::from_bits((x >> 16) as u16);
        }

        // Extract IEEE754 components
        let sign = x & 0x80000000u32;
        let exp  = x & 0x7F800000u32;
        let man  = x & 0x007FFFFFu32;

        // Denormals will underflow, so return signed zero
        if exp == 0 {
            return f16::from_bits(x as u16);
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7F800000u32 {
            // A mantissa of zero is a signed Infinity
            if man == 0 {
                return f16::from_bits(((x >> 16) | 0x7C00u32) as u16);
            }
            // Otherwise, this is NaN
            return f16::from_bits(0xFE00u16);
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = (exp >> 23) - 127;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return f16::from_bits((half_sign | 0x7C00u32) as u16);
        }

        // Check for underflow
        if half_sign <= 0 {
            // Check mantissa for what we can do
            if 14 - half_exp > 24 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return f16::from_bits(half_sign as u16);
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x00800000u32;
            let mut half_man = man >> (14 - half_exp);
            // Check for rounding
            if (man >> (13 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for denormals
            return f16::from_bits((half_sign | half_man) as u16);
        }

        // Rebias the exponent
        let half_exp = half_exp << 10;
        let half_man = man >> 13;
        // Check for rounding
        if man & 0x00001000u32 != 0 {
            // Round it
            f16::from_bits(((half_sign | half_exp | half_man) + 1) as u16)
        } else {
            f16::from_bits((half_sign | half_exp | half_man) as u16)
        }
    }

    pub fn from_f64(value: f64) -> f16 {
        // Convert to raw bytes, truncating the last 32-bits of mantissa; that precision will always
        // be lost on half-precision.
        let val: u64 = unsafe { mem::transmute(value) };
        let x = (val >> 32) as u32;

        // Check for signed zero
        if x & 0x7FFFFFFFu32 == 0 {
            return f16::from_bits((x >> 16) as u16);
        }

        // Extract IEEE754 components
        let sign = x & 0x80000000u32;
        let exp  = x & 0x7FF00000u32;
        let man  = x & 0x000FFFFFu32;

        // Denormals will underflow, so return signed zero
        if exp == 0 {
            return f16::from_bits(x as u16);
        }

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7FF00000u32 {
            // A mantissa of zero is a signed Infinity
            if man == 0 {
                return f16::from_bits(((x >> 16) | 0x7C00u32) as u16);
            }
            // Otherwise, this is NaN
            return f16::from_bits(0xFE00u16);
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = (exp >> 20) - 1023;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return f16::from_bits((half_sign | 0x7C00u32) as u16);
        }

        // Check for underflow
        if half_sign <= 0 {
            // Check mantissa for what we can do
            if 10 - half_exp > 21 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return f16::from_bits(half_sign as u16);
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x00100000u32;
            let mut half_man = man >> (11 - half_exp);
            // Check for rounding
            if (man >> (10 - half_exp)) & 0x1u32 != 0 {
                half_man += 1;
            }
            // No exponent for denormals
            return f16::from_bits((half_sign | half_man) as u16);
        }

        // Rebias the exponent
        let half_exp = half_exp << 10;
        let half_man = man >> 10;
        // Check for rounding
        if man & 0x00000200u32 != 0 {
            // Round it
            f16::from_bits(((half_sign | half_exp | half_man) + 1) as u16)
        } else {
            f16::from_bits((half_sign | half_exp | half_man) as u16)
        }
    }

    pub fn as_bits(&self) -> u16 {
        self.0
    }

    pub fn to_f32(&self) -> f32 {
        // Check for signed zero
        if self.0 & 0x7FFFu16 == 0 {
            return unsafe { mem::transmute((self.0 as u32) << 16) };
        }

        let half_sign = (self.0 & 0x8000u16) as u32;
        let half_exp  = (self.0 & 0x7C00u16) as u32;
        let half_man  = (self.0 & 0x03FFu16) as u32;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u32 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return unsafe { mem::transmute((half_sign << 16) | 0x7F800000u32) };
            } else {
                // NaN, only 1st mantissa bit is set
                return unsafe { mem::transmute(0xFFC00000u32) };
            }
        }

        // Calculate single-precision components with adjusted exponent
        let sign = half_sign << 16;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i32) >> 10) - 15;
        let man = (half_man & 0x03FFu32) << 13;

        // Check for denormals, which will be normalized by adjusting exponent
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

    pub fn to_f64(&self) -> f64 {
        // Check for signed zero
        if self.0 & 0x7FFFu16 == 0 {
            return unsafe { mem::transmute((self.0 as u64) << 48) };
        }

        let half_sign = (self.0 & 0x8000u16) as u64;
        let half_exp  = (self.0 & 0x7C00u16) as u64;
        let half_man  = (self.0 & 0x03FFu16) as u64;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u64 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                return unsafe { mem::transmute((half_sign << 48) | 0x7FF0000000000000u64) };
            } else {
                // NaN, only 1st mantissa bit is set
                return unsafe { mem::transmute(0xFFF8000000000000u64) };
            }
        }

        // Calculate double-precision components with adjusted exponent
        let sign = half_sign << 48;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i64) >> 10) - 15;
        let man = (half_man & 0x03FFu64) << 42;

        // Check for denormals, which will be normalized by adjusting exponent
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

//
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
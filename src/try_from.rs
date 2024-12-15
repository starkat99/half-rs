// Try to convert a value from a wider type, only converting
// the value if it can be losslessly converted, otherwise returning none.
macro_rules! try_from_lossless {
    (
        value =>
        $value:ident,half =>
        $half:ident,full =>
        $full:ident,half_bits =>
        $half_bits:ident,full_bits =>
        $full_bits:ident,to_half =>
        $to_half:ident
    ) => {{
        // let's use `f16` and `f64` as an example:
        // the `f64` is broken down into the following components:
        // - sign: 1
        // - exp:   111111111110000000000000000000000000000000000000000000000000000
        // - mant:             1111111111111111111111111111111111111111111111111111
        //
        // value is stored as `2^n` where the lowest bit of the exp is the implicit
        // hidden bit, that is, the `1`, while the top bit of `mantissa` is `1/2`,
        // then `1/4`, etc. for an `f16`, we then have:
        // - sign: 1
        // - exp:   11111
        // - mant:             1111111111
        // or the bottom 42 bits are truncated during the conversion, if the exponents
        // are in range. we only need to consider special cases, that is, subnormal
        // floats, where all exponent bits are 0, if both types have the same number of
        // exponent bits (`f32` to `bf16`). so only if the bottom `52 - 10` bits are
        // `0`, then it has a lossless conversion
        //
        // we do have special cases for non-finite values, NaN and +/- infinity. since
        // a NaN is still a NaN no matter the lower `51` bits in the longer type, if
        // we can ignore the result no matter the lower bits.
        let bits: $full_bits = unsafe { core::mem::transmute($value) };

        // get our masks and extract the IEEE754 components.
        const FULL_MANTISSA_BITS: u32 = <$full>::MANTISSA_DIGITS - 1;
        const FULL_SIGN_MASK: $full_bits = 1 << (<$full_bits>::BITS - 1);
        const FULL_EXPONENT_MASK: $full_bits =
            (<$full>::MAX_EXP as $full_bits * 2 - 1) << FULL_MANTISSA_BITS;
        const FULL_MANTISSA_MASK: $full_bits = (1 << FULL_MANTISSA_BITS) - 1;
        let full_sign = bits & FULL_SIGN_MASK;
        let full_exp = bits & FULL_EXPONENT_MASK;
        let full_mant = bits & FULL_MANTISSA_MASK;

        const HALF_MANTISSA_BITS: u32 = <$half>::MANTISSA_DIGITS - 1;
        const HALF_EXPONENT_MASK: $half_bits =
            (<$half>::MAX_EXP as $half_bits * 2 - 1) << HALF_MANTISSA_BITS;
        let sign_shift = <$full_bits>::BITS - <$half_bits>::BITS;
        let half_sign = (full_sign >> sign_shift) as $half_bits;

        // we use the number of bits without the hidden bit.
        // we want to know the number of bits truncated and a mask for
        // all bits that could be truncated.
        const TRUNCATED_BITS: u32 = FULL_MANTISSA_BITS - HALF_MANTISSA_BITS;

        // check for if we have a special (non-finite) number
        if full_exp == FULL_EXPONENT_MASK {
            let half_exp = HALF_EXPONENT_MASK;
            let half_mant = (full_mant >> TRUNCATED_BITS) as $half_bits;
            return Some($half(half_sign | half_exp | half_mant));
        }

        // check for zero, which would otherwise underflow
        if (bits & !FULL_SIGN_MASK) == 0 {
            return Some($half(half_sign));
        }

        // need to get our unbiased exponent. exponents are stored with
        // the value as (2^exp - (2^(expbits-1) - 1)`. the max, unbiased
        // exp for `bf16` is `127` and the min non-denormal one is `-126`.
        // we need the hidden bit in this biased exp.
        const FULL_BIAS: i32 = <$full>::MAX_EXP - 1;
        let full_biased = (full_exp >> FULL_MANTISSA_BITS) as i32;
        let full_unbiased = full_biased - FULL_BIAS;

        // now we need to know if our exponent is in the range. our range is from
        // if your small exp is valid for our float, that is, unbiased it's in
        // the range `2 - 2^(expbits-1)` or `1 - bias` for a normal float
        // (biased exp `>= 1`), but a denormal float works so we want
        // `1 - bias`. Our max exp finite is `2^(expbits-1) - 1` or `bias`.
        // all special values always are valid, so we also accept when all
        // the exponent bits are set. we have a special case: when the two
        // exponents are the same number of bits: then it's **ALWAYS** valid.
        //
        // but this still needs to consider denormal values, or where we have
        // no exp bits
        const HALF_BIAS: i32 = <$half>::MAX_EXP - 1;
        const HALF_MIN_EXP: i32 = 1 - HALF_BIAS;
        const FULL_EXP_BITS: u32 = <$full_bits>::BITS - FULL_MANTISSA_BITS - 1;
        const HALF_EXP_BITS: u32 = <$half_bits>::BITS - HALF_MANTISSA_BITS - 1;
        const HALF_MAX_EXP: i32 = HALF_BIAS;
        const HALF_MIN_DENORMAL_EXP: i32 = HALF_MIN_EXP - HALF_MANTISSA_BITS as i32;
        let exp_in_range = if FULL_EXP_BITS == HALF_EXP_BITS {
            true
        } else {
            full_unbiased >= HALF_MIN_DENORMAL_EXP && full_unbiased <= HALF_MAX_EXP
        };
        if !exp_in_range {
            return None;
        }

        // get if we have any truncated bits, otherwise, we have an exact result
        let half_biased = full_unbiased + HALF_BIAS;
        let is_denormal = half_biased <= 0;
        let truncated_bits = if is_denormal {
            // NOTE: This needs an extra bit for what was formerly the hidden bit
            (TRUNCATED_BITS as i32 - half_biased + 1) as u32
        } else {
            TRUNCATED_BITS
        };
        let truncated_mask: $full_bits = (1 << truncated_bits) - 1;
        if bits & truncated_mask != 0 {
            return None;
        }

        // now we need to reassemble our float components. remember if we have
        // a denormal float in the result we need to move our implicit hidden
        // bit out.
        let full_hidden_bit: $full_bits = 1 << FULL_MANTISSA_BITS;
        let (half_mant, half_exp) = if is_denormal {
            let half_mant = ((full_mant | full_hidden_bit) >> truncated_bits) as $half_bits;
            (half_mant, 0)
        } else {
            let half_mant = (full_mant >> truncated_bits) as $half_bits;
            let half_exp = (half_biased as $half_bits) << HALF_MANTISSA_BITS;
            (half_mant, half_exp)
        };

        Some($half(half_sign | half_exp | half_mant))
    }};
}

pub(crate) use try_from_lossless;

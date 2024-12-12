// https://doc.rust-lang.org/std/primitive.u16.html#method.leading_zeros

#[cfg(not(any(all(
    target_arch = "spirv",
    not(all(
        target_feature = "IntegerFunctions2INTEL",
        target_feature = "SPV_INTEL_shader_integer_functions2"
    ))
))))]
#[inline]
pub(crate) const fn leading_zeros_u16(x: u16) -> u32 {
    x.leading_zeros()
}

#[cfg(all(
    target_arch = "spirv",
    not(all(
        target_feature = "IntegerFunctions2INTEL",
        target_feature = "SPV_INTEL_shader_integer_functions2"
    ))
))]
#[inline]
pub(crate) const fn leading_zeros_u16(x: u16) -> u32 {
    leading_zeros_u16_fallback(x)
}

#[cfg(any(
    test,
    all(
        target_arch = "spirv",
        not(all(
            target_feature = "IntegerFunctions2INTEL",
            target_feature = "SPV_INTEL_shader_integer_functions2"
        ))
    )
))]
#[inline]
const fn leading_zeros_u16_fallback(x: u16) -> u32 {
    leading_zeros_u16_fallback_impl(x)
}

// NOTE: We manually unroll this to avoid dependencies.
// On most (all?) architectures Rust is smart enough
// to unroll it if we use a while loop.
macro_rules! unrolled_leading_zeros {
    ($x:ident, $c:ident, $msb:ident) => {{
        if $x & $msb == 0 {
            $c += 1;
        } else {
            return $c;
        }
        $x <<= 1;
    }};
}

// NOTE: Exposed for testing.
#[inline]
#[allow(dead_code)]
const fn leading_zeros_u16_fallback_impl(mut x: u16) -> u32 {
    let mut c = 0;
    let msb = 1 << 15;
    // NOTE: Crunchy isn't required since it's only required
    // if we use the loop variable, so we just use the first
    // 14 and then use the final one outside.
    unrolled_leading_zeros!(x, c, msb); // 1
    unrolled_leading_zeros!(x, c, msb); // 2
    unrolled_leading_zeros!(x, c, msb); // 3
    unrolled_leading_zeros!(x, c, msb); // 4
    unrolled_leading_zeros!(x, c, msb); // 5
    unrolled_leading_zeros!(x, c, msb); // 6
    unrolled_leading_zeros!(x, c, msb); // 7
    unrolled_leading_zeros!(x, c, msb); // 8
    unrolled_leading_zeros!(x, c, msb); // 9
    unrolled_leading_zeros!(x, c, msb); // 10
    unrolled_leading_zeros!(x, c, msb); // 11
    unrolled_leading_zeros!(x, c, msb); // 12
    unrolled_leading_zeros!(x, c, msb); // 13
    unrolled_leading_zeros!(x, c, msb); // 14
    if x & msb == 0 {
        c += 1;
    }
    c
}

#[cfg(test)]
mod test {

    #[test]
    fn leading_zeros_u16_fallback() {
        for x in [44, 97, 304, 1179, 23571] {
            assert_eq!(super::leading_zeros_u16_fallback(x), x.leading_zeros());
            assert_eq!(super::leading_zeros_u16_fallback_impl(x), x.leading_zeros());
        }
    }
}

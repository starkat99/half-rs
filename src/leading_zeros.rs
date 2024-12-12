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
const fn leading_zeros_u16_fallback(mut x: u16) -> u32 {
    let mut c = 0;
    let msb = 1 << 15;
    // NOTE: Crunchy isn't required since it's only required
    // if we use the loop variable, so we just use the first
    // 14 and then use the final one outside.
    for i in 0..=14 {
        if x & msb == 0 {
            c += 1;
        } else {
            return c;
        }
        x <<= 1;
    }
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
        }
    }
}

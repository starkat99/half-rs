use float16::prelude::*;
use quickcheck_macros::quickcheck;

#[test]
fn formatting() {
    let f = f16::from_f32(0.1152344);

    assert_eq!(format!("{:.3}", f), "0.115");
    assert_eq!(format!("{:.4}", f), "0.1152");
    assert_eq!(format!("{:+.4}", f), "+0.1152");
    assert_eq!(format!("{:>+10.4}", f), "   +0.1152");

    assert_eq!(format!("{:.3?}", f), "0.115");
    assert_eq!(format!("{:.4?}", f), "0.1152");
    assert_eq!(format!("{:+.4?}", f), "+0.1152");
    assert_eq!(format!("{:>+10.4?}", f), "   +0.1152");
}

#[quickcheck]
fn qc_roundtrip_f16_f32_is_identity(bits: u16) -> bool {
    let f = f16::from_bits(bits);
    let roundtrip = f16::from_f32(f.to_f32());
    if f.is_nan() {
        roundtrip.is_nan() && f.is_sign_negative() == roundtrip.is_sign_negative()
    } else {
        f.to_bits() == roundtrip.to_bits()
    }
}

#[quickcheck]
fn qc_roundtrip_f16_f64_is_identity(bits: u16) -> bool {
    let f = f16::from_bits(bits);
    let roundtrip = f16::from_f64(f.to_f64());
    if f.is_nan() {
        roundtrip.is_nan() && f.is_sign_negative() == roundtrip.is_sign_negative()
    } else {
        f.to_bits() == roundtrip.to_bits()
    }
}

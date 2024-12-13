fn main() {
    let version = rustc_version::version().unwrap();

    if version >= rustc_version::Version::parse("1.63.0").unwrap() {
        println!("cargo:rustc-cfg=has_aarch64_intrinsics");
    }

    if version >= rustc_version::Version::parse("1.70.0").unwrap() {
        println!("cargo:rustc-cfg=has_x86_intrinsics");
    }
}

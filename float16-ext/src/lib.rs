// TODO: Here

#[cfg(not(target_arch = "spirv"))]
pub mod slice;
#[cfg(feature = "alloc")]
pub mod vec;

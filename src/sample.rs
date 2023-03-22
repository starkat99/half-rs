use rand_distr::{uniform::UniformFloat, Distribution};

macro_rules! impl_distribution_via_f32 {
    ($Ty:ty, $Distr:ty) => {
        impl Distribution<$Ty> for $Distr {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $Ty {
                <$Ty>::from_f32(<Self as Distribution<f32>>::sample(self, rng))
            }
        }
    };
}

impl_distribution_via_f32!(crate::f16, rand_distr::Standard);
impl_distribution_via_f32!(crate::f16, rand_distr::StandardNormal);
impl_distribution_via_f32!(crate::f16, rand_distr::Exp1);
impl_distribution_via_f32!(crate::f16, rand_distr::Open01);
impl_distribution_via_f32!(crate::f16, rand_distr::OpenClosed01);

impl_distribution_via_f32!(crate::bf16, rand_distr::Standard);
impl_distribution_via_f32!(crate::bf16, rand_distr::StandardNormal);
impl_distribution_via_f32!(crate::bf16, rand_distr::Exp1);
impl_distribution_via_f32!(crate::bf16, rand_distr::Open01);
impl_distribution_via_f32!(crate::bf16, rand_distr::OpenClosed01);

#[derive(Debug, Clone, Copy)]
pub struct Float16Sampler(UniformFloat<f32>);

impl rand_distr::uniform::SampleUniform for crate::f16 {
    type Sampler = Float16Sampler;
}

impl rand_distr::uniform::UniformSampler for Float16Sampler {
    type X = crate::f16;
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::new(
            low.borrow().to_f32(),
            high.borrow().to_f32(),
        ))
    }
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::new_inclusive(
            low.borrow().to_f32(),
            high.borrow().to_f32(),
        ))
    }
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        crate::f16::from_f32(self.0.sample(rng))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BFloat16Sampler(UniformFloat<f32>);

impl rand_distr::uniform::SampleUniform for crate::bf16 {
    type Sampler = BFloat16Sampler;
}

impl rand_distr::uniform::UniformSampler for BFloat16Sampler {
    type X = crate::bf16;
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::new(
            low.borrow().to_f32(),
            high.borrow().to_f32(),
        ))
    }
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::new_inclusive(
            low.borrow().to_f32(),
            high.borrow().to_f32(),
        ))
    }
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        crate::bf16::from_f32(self.0.sample(rng))
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    #[test]
    fn test_sample_f16() {
        let mut rng = thread_rng();
        let _: crate::f16 = rng.sample(rand_distr::Standard);
        let _: crate::f16 = rng.sample(rand_distr::StandardNormal);
        let _: crate::f16 = rng.sample(
            rand_distr::Normal::new(crate::f16::from_f32(0.0), crate::f16::from_f32(1.0)).unwrap(),
        );
        let _: crate::f16 = rng.sample(rand_distr::Uniform::new(
            crate::f16::from_f32(0.0),
            crate::f16::from_f32(1.0),
        ));
    }

    #[test]
    fn test_sample_bf16() {
        let mut rng = thread_rng();
        let _: crate::bf16 = rng.sample(rand_distr::Standard);
        let _: crate::bf16 = rng.sample(rand_distr::StandardNormal);
        let _: crate::bf16 = rng.sample(
            rand_distr::Normal::new(crate::bf16::from_f32(0.0), crate::bf16::from_f32(1.0))
                .unwrap(),
        );
        let _: crate::bf16 = rng.sample(rand_distr::Uniform::new(
            crate::bf16::from_f32(0.0),
            crate::bf16::from_f32(1.0),
        ));
    }
}

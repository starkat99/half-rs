#[macro_use]
extern crate criterion;
extern crate half;

use criterion::{Bencher, Criterion};
use half::{
    bfloat::{self, bf16},
    consts, f16,
};
use std::{f32, f64};

fn bench_f32_to_f16(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "f16::from_f32",
        |b: &mut Bencher, i: &f32| b.iter(|| f16::from_f32(*i)),
        vec![
            0.,
            -0.,
            1.,
            -1.,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            f32::consts::E,
            f32::consts::PI,
        ],
    );
}

fn bench_f64_to_f16(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "f16::from_f64",
        |b: &mut Bencher, i: &f64| b.iter(|| f16::from_f64(*i)),
        vec![
            0.,
            -0.,
            1.,
            -1.,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            f64::consts::E,
            f64::consts::PI,
        ],
    );
}

fn bench_f16_to_f32(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "f16::to_f32",
        |b: &mut Bencher, i: &f16| b.iter(|| i.to_f32()),
        vec![
            f16::from_f32(0.),
            f16::from_f32(-0.),
            f16::from_f32(1.),
            f16::from_f32(-1.),
            consts::MIN,
            consts::MAX,
            consts::MIN_POSITIVE,
            consts::NEG_INFINITY,
            consts::INFINITY,
            consts::NAN,
            consts::E,
            consts::PI,
        ],
    );
}

fn bench_f16_to_f64(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "f16::to_f64",
        |b: &mut Bencher, i: &f16| b.iter(|| i.to_f64()),
        vec![
            f16::from_f32(0.),
            f16::from_f32(-0.),
            f16::from_f32(1.),
            f16::from_f32(-1.),
            consts::MIN,
            consts::MAX,
            consts::MIN_POSITIVE,
            consts::NEG_INFINITY,
            consts::INFINITY,
            consts::NAN,
            consts::E,
            consts::PI,
        ],
    );
}

fn bench_f32_to_bf16(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "bf16::from_f32",
        |b: &mut Bencher, i: &f32| b.iter(|| bf16::from_f32(*i)),
        vec![
            0.,
            -0.,
            1.,
            -1.,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            f32::consts::E,
            f32::consts::PI,
        ],
    );
}

fn bench_f64_to_bf16(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "bf16::from_f64",
        |b: &mut Bencher, i: &f64| b.iter(|| bf16::from_f64(*i)),
        vec![
            0.,
            -0.,
            1.,
            -1.,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            f64::consts::E,
            f64::consts::PI,
        ],
    );
}

fn bench_bf16_to_f32(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "bf16::to_f32",
        |b: &mut Bencher, i: &bf16| b.iter(|| i.to_f32()),
        vec![
            bf16::from_f32(0.),
            bf16::from_f32(-0.),
            bf16::from_f32(1.),
            bf16::from_f32(-1.),
            bfloat::consts::MIN,
            bfloat::consts::MAX,
            bfloat::consts::MIN_POSITIVE,
            bfloat::consts::NEG_INFINITY,
            bfloat::consts::INFINITY,
            bfloat::consts::NAN,
            bfloat::consts::E,
            bfloat::consts::PI,
        ],
    );
}

fn bench_bf16_to_f64(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "bf16::to_f64",
        |b: &mut Bencher, i: &bf16| b.iter(|| i.to_f64()),
        vec![
            bf16::from_f32(0.),
            bf16::from_f32(-0.),
            bf16::from_f32(1.),
            bf16::from_f32(-1.),
            bfloat::consts::MIN,
            bfloat::consts::MAX,
            bfloat::consts::MIN_POSITIVE,
            bfloat::consts::NEG_INFINITY,
            bfloat::consts::INFINITY,
            bfloat::consts::NAN,
            bfloat::consts::E,
            bfloat::consts::PI,
        ],
    );
}

criterion_group!(
    f16_sisd,
    bench_f32_to_f16,
    bench_f64_to_f16,
    bench_f16_to_f32,
    bench_f16_to_f64
);
criterion_group!(
    bf16_sisd,
    bench_f32_to_bf16,
    bench_f64_to_bf16,
    bench_bf16_to_f32,
    bench_bf16_to_f64
);
criterion_main!(f16_sisd, bf16_sisd);

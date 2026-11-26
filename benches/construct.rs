#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal)]

use bencher::{benchmark_group, benchmark_main, Bencher};

use ndarray::prelude::*;

fn default_f64(bench: &mut Bencher)
{
    bench.iter(|| Array::<f64, _>::default((128, 128)))
}

fn zeros_f64(bench: &mut Bencher)
{
    bench.iter(|| Array::<f64, _>::zeros((128, 128)))
}

#[cfg(feature = "std")]
fn map_regular(bench: &mut Bencher)
{
    let a = Array::linspace(0.0..=127.0, 128)
        .into_shape_with_order((8, 16))
        .unwrap();
    bench.iter(|| a.map(|&x| 2. * x));
}

#[cfg(feature = "std")]
fn map_stride(bench: &mut Bencher)
{
    let a = Array::linspace(0.0..=127.0, 256)
        .into_shape_with_order((8, 32))
        .unwrap();
    let av = a.slice(s![.., ..;2]);
    bench.iter(|| av.map(|&x| 2. * x));
}

benchmark_group!(
    benches,
    default_f64,
    zeros_f64,
    map_regular,
    map_stride,
);
benchmark_main!(benches);

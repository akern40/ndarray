#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal)]

use bencher::{benchmark_group, benchmark_main, Bencher};
use std::hint::black_box;

use ndarray::prelude::*;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[cfg(feature = "std")]
fn map_regular(bench: &mut Bencher)
{
    let a = Array::linspace(0.0..=127.0, N)
        .into_shape_with_order((X, Y))
        .unwrap();
    bench.iter(|| a.map(|&x| 2. * x));
}

pub fn double_array(mut a: ArrayViewMut2<'_, f64>)
{
    a *= 2.0;
}

#[cfg(feature = "std")]
fn map_stride_double_f64(bench: &mut Bencher)
{
    let mut a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let mut av = a.slice_mut(s![.., ..;2]);
    bench.iter(|| {
        double_array(av.view_mut());
    });
}

#[cfg(feature = "std")]
fn map_stride_f64(bench: &mut Bencher)
{
    let a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let av = a.slice(s![.., ..;2]);
    bench.iter(|| av.map(|&x| 2. * x));
}

#[cfg(feature = "std")]
fn map_stride_u32(bench: &mut Bencher)
{
    let a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let b = a.mapv(|x| x as u32);
    let av = b.slice(s![.., ..;2]);
    bench.iter(|| av.map(|&x| 2 * x));
}

#[cfg(feature = "std")]
fn fold_axis(bench: &mut Bencher)
{
    let a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    bench.iter(|| a.fold_axis(Axis(0), 0., |&acc, &elt| acc + elt));
}

const MA: usize = 64;
const MASZ: usize = MA * MA;

fn map_axis_0(bench: &mut Bencher)
{
    let a = Array::from_iter(0..MASZ as i32)
        .into_shape_with_order([MA, MA])
        .unwrap();
    bench.iter(|| a.map_axis(Axis(0), black_box));
}

fn map_axis_1(bench: &mut Bencher)
{
    let a = Array::from_iter(0..MASZ as i32)
        .into_shape_with_order([MA, MA])
        .unwrap();
    bench.iter(|| a.map_axis(Axis(1), black_box));
}

benchmark_group!(
    benches,
    map_regular,
    map_stride_double_f64,
    map_stride_f64,
    map_stride_u32,
    fold_axis,
    map_axis_0,
    map_axis_1,
);
benchmark_main!(benches);

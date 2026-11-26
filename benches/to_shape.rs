use bencher::{benchmark_group, benchmark_main, Bencher};

use ndarray::prelude::*;
use ndarray::Order;

fn to_shape2_1(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape(4 * 5).unwrap());
}

fn to_shape2_2_same(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape((4, 5)).unwrap());
}

fn to_shape2_2_flip(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape((5, 4)).unwrap());
}

fn to_shape2_3(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape((2, 5, 2)).unwrap());
}

fn to_shape3_1(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape(3 * 4 * 5).unwrap());
}

fn to_shape3_2_order(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape((12, 5)).unwrap());
}

fn to_shape3_2_outoforder(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape((4, 15)).unwrap());
}

fn to_shape3_3c(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape((3, 4, 5)).unwrap());
}

fn to_shape3_3f(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5).f());
    let view = a.view();
    bench.iter(|| view.to_shape(((3, 4, 5), Order::F)).unwrap());
}

fn to_shape3_4c(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| view.to_shape(((2, 3, 2, 5), Order::C)).unwrap());
}

fn to_shape3_4f(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((3, 4, 5).f());
    let view = a.view();
    bench.iter(|| view.to_shape(((2, 3, 2, 5), Order::F)).unwrap());
}

benchmark_group!(
    benches,
    to_shape2_1,
    to_shape2_2_same,
    to_shape2_2_flip,
    to_shape2_3,
    to_shape3_1,
    to_shape3_2_order,
    to_shape3_2_outoforder,
    to_shape3_3c,
    to_shape3_3f,
    to_shape3_4c,
    to_shape3_4f,
);
benchmark_main!(benches);

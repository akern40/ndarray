use bencher::{benchmark_group, benchmark_main, Bencher};

use ndarray::prelude::*;

fn push_reserve(bench: &mut Bencher)
{
    let ones: Array<f32, _> = array![1f32];
    bench.iter(|| {
        let mut a: Array<f32, Ix1> = array![];
        a.reserve(Axis(0), 100).unwrap();
        for _ in 0..100 {
            a.append(Axis(0), ones.view()).unwrap();
        }
    });
}

fn push_no_reserve(bench: &mut Bencher)
{
    let ones: Array<f32, _> = array![1f32];
    bench.iter(|| {
        let mut a: Array<f32, Ix1> = array![];
        for _ in 0..100 {
            a.append(Axis(0), ones.view()).unwrap();
        }
    });
}

benchmark_group!(
    benches,
    push_reserve,
    push_no_reserve,
);
benchmark_main!(benches);

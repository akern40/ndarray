use bencher::{benchmark_group, benchmark_main, Bencher};
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rand_distr::Uniform;

fn uniform_f32(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Uniform::new(-1f32, 1.).unwrap()));
}

fn norm_f32(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(0f32, 1.).unwrap()));
}

fn norm_f64(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(0f64, 1.).unwrap()));
}

benchmark_group!(benches, uniform_f32, norm_f32, norm_f64);
benchmark_main!(benches);

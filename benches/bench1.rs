#![allow(unused_imports)]
#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal)]

use bencher::{benchmark_group, benchmark_main, Bencher};
use std::mem::MaybeUninit;

use ndarray::{arr0, arr1, arr2, azip, s};
use ndarray::{Array, Array1, Array2, Axis, Ix, Zip};
use ndarray::{Array3, Array4, ShapeBuilder};
use ndarray::{Ix1, Ix2, Ix3, Ix5, IxDyn};

use std::hint::black_box;

fn iter_sum_1d_regular(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros(64 * 64);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

fn iter_sum_1d_raw(bench: &mut Bencher)
{
    // this is autovectorized to death (= great performance)
    let a = Array::<i32, _>::zeros(64 * 64);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.as_slice_memory_order().unwrap() {
            sum += elt;
        }
        sum
    });
}

fn iter_sum_2d_regular(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

fn iter_sum_2d_by_row(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for row in a.rows() {
            for &elt in row {
                sum += elt;
            }
        }
        sum
    });
}

fn iter_sum_2d_raw(bench: &mut Bencher)
{
    // this is autovectorized to death (= great performance)
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.as_slice_memory_order().unwrap() {
            sum += elt;
        }
        sum
    });
}

fn iter_sum_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

fn iter_sum_2d_cutout_by_row(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for row in 0..a.shape()[0] {
            for &elt in a.row(row) {
                sum += elt;
            }
        }
        sum
    });
}

fn iter_sum_2d_cutout_outer_iter(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for row in a.rows() {
            for &elt in row {
                sum += elt;
            }
        }
        sum
    });
}

fn iter_sum_2d_transpose_regular(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

fn iter_sum_2d_transpose_by_row(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for row in 0..a.shape()[0] {
            for &elt in a.row(row) {
                sum += elt;
            }
        }
        sum
    });
}

fn sum_2d_regular(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| a.sum());
}

fn sum_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| a.sum());
}

fn sum_2d_float(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let a = black_box(a.view());
    bench.iter(|| a.sum());
}

fn sum_2d_float_cutout(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| a.sum());
}

fn sum_2d_float_t_cutout(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]).reversed_axes();
    let a = black_box(av);
    bench.iter(|| a.sum());
}

fn fold_sum_i32_2d_regular(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    bench.iter(|| a.fold(0, |acc, &x| acc + x));
}

fn fold_sum_i32_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| a.fold(0, |acc, &x| acc + x));
}

fn fold_sum_i32_2d_stride(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 128));
    let av = a.slice(s![.., ..;2]);
    let a = black_box(av);
    bench.iter(|| a.fold(0, |acc, &x| acc + x));
}

fn fold_sum_i32_2d_transpose(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = a.t();
    bench.iter(|| a.fold(0, |acc, &x| acc + x));
}

fn fold_sum_i32_2d_cutout_transpose(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let mut av = a.slice(s![1..-1, 1..-1]);
    av.swap_axes(0, 1);
    let a = black_box(av);
    bench.iter(|| a.fold(0, |acc, &x| acc + x));
}

const ADD2DSZ: usize = 64;

fn add_2d_regular(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn add_2d_zip(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| {
        Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
    });
}

fn add_2d_alloc_plus(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| &a + &b);
}

fn add_2d_alloc_zip_uninit(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| unsafe {
        let mut c = Array::<i32, _>::uninit(a.dim());
        azip!((&a in &a, &b in &b, c in c.raw_view_mut().cast::<i32>())
            c.write(a + b)
        );
        c
    });
}

fn add_2d_alloc_zip_collect(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| Zip::from(&a).and(&b).map_collect(|&x, &y| x + y));
}

fn vec_string_collect(bench: &mut Bencher)
{
    let v = vec![""; 10240];
    bench.iter(|| v.iter().map(|s| s.to_owned()).collect::<Vec<_>>());
}

fn array_string_collect(bench: &mut Bencher)
{
    let v = Array::from(vec![""; 10240]);
    bench.iter(|| Zip::from(&v).map_collect(|s| s.to_owned()));
}

fn vec_f64_collect(bench: &mut Bencher)
{
    let v = vec![1.; 10240];
    bench.iter(|| v.iter().map(|s| s + 1.).collect::<Vec<_>>());
}

fn array_f64_collect(bench: &mut Bencher)
{
    let v = Array::from(vec![1.; 10240]);
    bench.iter(|| Zip::from(&v).map_collect(|s| s + 1.));
}

fn add_2d_assign_ops(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    bench.iter(|| {
        let mut x = a.view_mut();
        x += &bv;
        black_box(x);
    });
}

fn add_2d_cutout(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ + 2, ADD2DSZ + 2));
    let mut acut = a.slice_mut(s![1..-1, 1..-1]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    bench.iter(|| {
        acut += &bv;
    });
}

fn add_2d_zip_cutout(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ + 2, ADD2DSZ + 2));
    let mut acut = a.slice_mut(s![1..-1, 1..-1]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| {
        Zip::from(&mut acut).and(&b).for_each(|a, &b| *a += b);
    });
}

#[allow(clippy::identity_op)]
fn add_2d_cutouts_by_4(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let b = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let chunksz = (4, 4);
    bench.iter(|| {
        Zip::from(a.exact_chunks_mut(chunksz))
            .and(b.exact_chunks(chunksz))
            .for_each(|mut a, b| a += &b);
    });
}

#[allow(clippy::identity_op)]
fn add_2d_cutouts_by_16(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let b = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let chunksz = (16, 16);
    bench.iter(|| {
        Zip::from(a.exact_chunks_mut(chunksz))
            .and(b.exact_chunks(chunksz))
            .for_each(|mut a, b| a += &b);
    });
}

#[allow(clippy::identity_op)]
fn add_2d_cutouts_by_32(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let b = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let chunksz = (32, 32);
    bench.iter(|| {
        Zip::from(a.exact_chunks_mut(chunksz))
            .and(b.exact_chunks(chunksz))
            .for_each(|mut a, b| a += &b);
    });
}

fn add_2d_broadcast_1_to_2(bench: &mut Bencher)
{
    let mut a = Array2::<i32>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array1::<i32>::zeros(ADD2DSZ);
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn add_2d_broadcast_0_to_2(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros(());
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn scalar_toowned(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    bench.iter(|| a.to_owned());
}

fn scalar_add_1(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    bench.iter(|| &a + n);
}

fn scalar_add_2(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    bench.iter(|| n + &a);
}

fn scalar_add_strided_1(bench: &mut Bencher)
{
    let a = Array::from_shape_fn((64, 64 * 2), |(i, j)| (i * 64 + j) as f32).slice_move(s![.., ..;2]);
    let n = 1.;
    bench.iter(|| &a + n);
}

fn scalar_add_strided_2(bench: &mut Bencher)
{
    let a = Array::from_shape_fn((64, 64 * 2), |(i, j)| (i * 64 + j) as f32).slice_move(s![.., ..;2]);
    let n = 1.;
    bench.iter(|| n + &a);
}

fn scalar_sub_1(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    bench.iter(|| &a - n);
}

fn scalar_sub_2(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    bench.iter(|| n - &a);
}

// This is for comparison with add_2d_broadcast_0_to_2
fn add_2d_0_to_2_iadd_scalar(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let n = black_box(0);
    bench.iter(|| {
        a += n;
    });
}

fn add_2d_strided(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ * 2));
    let mut a = a.slice_mut(s![.., ..;2]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn add_2d_regular_dyn(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ][..]);
    let b = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ][..]);
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn add_2d_strided_dyn(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ * 2][..]);
    let mut a = a.slice_mut(s![.., ..;2]);
    let b = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ][..]);
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn add_2d_zip_strided(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ * 2));
    let mut a = a.slice_mut(s![.., ..;2]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| {
        Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
    });
}

fn add_2d_one_transposed(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| {
        a += &b;
    });
}

fn add_2d_zip_one_transposed(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| {
        Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
    });
}

fn add_2d_both_transposed(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    b.swap_axes(0, 1);
    bench.iter(|| {
        a += &b;
    });
}

fn add_2d_zip_both_transposed(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    b.swap_axes(0, 1);
    bench.iter(|| {
        Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
    });
}

fn add_2d_f32_regular(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

const ADD3DSZ: usize = 16;

fn add_3d_strided(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros((ADD3DSZ, ADD3DSZ, ADD3DSZ * 2));
    let mut a = a.slice_mut(s![.., .., ..;2]);
    let b = Array::<i32, _>::zeros(a.dim());
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

fn add_3d_strided_dyn(bench: &mut Bencher)
{
    let mut a = Array::<i32, _>::zeros(&[ADD3DSZ, ADD3DSZ, ADD3DSZ * 2][..]);
    let mut a = a.slice_mut(s![.., .., ..;2]);
    let b = Array::<i32, _>::zeros(a.dim());
    let bv = b.view();
    bench.iter(|| {
        a += &bv;
    });
}

const ADD1D_SIZE: usize = 64 * 64;

fn add_1d_regular(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros(ADD1D_SIZE);
    let b = Array::<f32, _>::zeros(a.dim());
    bench.iter(|| {
        a += &b;
    });
}

fn add_1d_strided(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros(ADD1D_SIZE * 2);
    let mut av = a.slice_mut(s![..;2]);
    let b = Array::<f32, _>::zeros(av.dim());
    bench.iter(|| {
        av += &b;
    });
}

fn iadd_scalar_2d_regular(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    bench.iter(|| {
        a += 1.;
    });
}

fn iadd_scalar_2d_strided(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ * 2));
    let mut a = a.slice_mut(s![.., ..;2]);
    bench.iter(|| {
        a += 1.;
    });
}

fn iadd_scalar_2d_regular_dyn(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros(vec![ADD2DSZ, ADD2DSZ]);
    bench.iter(|| {
        a += 1.;
    });
}

fn iadd_scalar_2d_strided_dyn(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros(vec![ADD2DSZ, ADD2DSZ * 2]);
    let mut a = a.slice_mut(s![.., ..;2]);
    bench.iter(|| {
        a += 1.;
    });
}

fn scaled_add_2d_f32_regular(bench: &mut Bencher)
{
    let mut av = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let scalar = std::f32::consts::PI;
    bench.iter(|| {
        av.scaled_add(scalar, &bv);
    });
}

fn assign_scalar_2d_corder(bench: &mut Bencher)
{
    let a = Array::zeros((ADD2DSZ, ADD2DSZ));
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(move || a.fill(s))
}

fn assign_scalar_2d_cutout(bench: &mut Bencher)
{
    let mut a = Array::zeros((66, 66));
    let a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(move || a.fill(s))
}

fn assign_scalar_2d_forder(bench: &mut Bencher)
{
    let mut a = Array::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(move || a.fill(s))
}

fn assign_zero_2d_corder(bench: &mut Bencher)
{
    let a = Array::zeros((ADD2DSZ, ADD2DSZ));
    let mut a = black_box(a);
    bench.iter(|| a.fill(0.))
}

fn assign_zero_2d_cutout(bench: &mut Bencher)
{
    let mut a = Array::zeros((66, 66));
    let a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    bench.iter(|| a.fill(0.))
}

fn assign_zero_2d_forder(bench: &mut Bencher)
{
    let mut a = Array::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    bench.iter(|| a.fill(0.))
}

fn bench_iter_diag(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    bench.iter(|| {
        for elt in a.diag() {
            black_box(elt);
        }
    })
}

fn bench_row_iter(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    let it = a.row(17);
    bench.iter(|| {
        for elt in it {
            black_box(elt);
        }
    })
}

fn bench_col_iter(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    let it = a.column(17);
    bench.iter(|| {
        for elt in it {
            black_box(elt);
        }
    })
}

macro_rules! mat_mul {
    ($modname:ident, $ty:ident, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use bencher::Bencher;
            use std::hint::black_box;
            use ndarray::Array;
            $(
            pub fn $name(bench: &mut Bencher)
            {
                let a = Array::<$ty, _>::zeros(($m, $n));
                let b = Array::<$ty, _>::zeros(($n, $k));
                let a = black_box(a.view());
                let b = black_box(b.view());
                bench.iter(|| a.dot(&b));
            }
            )+
        }

        // Registered with their module path so the f32, f64 and i32 variants get distinct names.
        benchmark_group!($modname, $($modname::$name),+);
    };
}

mat_mul! {mat_mul_f32, f32,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
    (mix10000, 128, 10000, 128)
}

mat_mul! {mat_mul_f64, f64,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
    (mix10000, 128, 10000, 128)
}

mat_mul! {mat_mul_i32, i32,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
}

fn create_iter_4d(bench: &mut Bencher)
{
    let mut a = Array::from_elem((4, 5, 3, 2), 1.0);
    a.swap_axes(0, 1);
    a.swap_axes(2, 1);
    let v = black_box(a.view());

    bench.iter(|| v.into_iter());
}

fn bench_to_owned_n(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((32, 32));
    bench.iter(|| a.to_owned());
}

fn bench_to_owned_t(bench: &mut Bencher)
{
    let mut a = Array::<f32, _>::zeros((32, 32));
    a.swap_axes(0, 1);
    bench.iter(|| a.to_owned());
}

fn bench_to_owned_strided(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((32, 64));
    let a = a.slice(s![.., ..;2]);
    bench.iter(|| a.to_owned());
}
fn equality_i32(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let b = Array::<i32, _>::zeros((64, 64));
    bench.iter(|| a == b);
}

fn equality_f32(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let b = Array::<f32, _>::zeros((64, 64));
    bench.iter(|| a == b);
}

fn equality_f32_mixorder(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let b = Array::<f32, _>::zeros((64, 64).f());
    bench.iter(|| a == b);
}

fn dot_f32_16(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(16);
    let b = Array::<f32, _>::zeros(16);
    bench.iter(|| a.dot(&b));
}

fn dot_f32_20(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(20);
    let b = Array::<f32, _>::zeros(20);
    bench.iter(|| a.dot(&b));
}

fn dot_f32_32(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(32);
    let b = Array::<f32, _>::zeros(32);
    bench.iter(|| a.dot(&b));
}

fn dot_f32_256(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(256);
    let b = Array::<f32, _>::zeros(256);
    bench.iter(|| a.dot(&b));
}

fn dot_f32_1024(bench: &mut Bencher)
{
    let av = Array::<f32, _>::zeros(1024);
    let bv = Array::<f32, _>::zeros(1024);
    bench.iter(|| av.dot(&bv));
}

fn dot_f32_10e6(bench: &mut Bencher)
{
    let n = 1_000_000;
    let av = Array::<f32, _>::zeros(n);
    let bv = Array::<f32, _>::zeros(n);
    bench.iter(|| av.dot(&bv));
}

fn dot_extended(bench: &mut Bencher)
{
    let m = 10;
    let n = 33;
    let k = 10;
    let av = Array::<f32, _>::zeros((m, n));
    let bv = Array::<f32, _>::zeros((n, k));
    let mut res = Array::<f32, _>::zeros((m, k));
    // make a manual simple matrix multiply to test
    bench.iter(|| {
        for i in 0..m {
            for j in 0..k {
                unsafe {
                    *res.uget_mut((i, j)) = av.row(i).dot(&bv.column(j));
                }
            }
        }
    })
}

const MEAN_SUM_N: usize = 127;

#[cfg(feature = "std")]
fn range_mat(m: Ix, n: Ix) -> Array2<f32>
{
    assert!(m * n != 0);
    Array::linspace(0.0..=(m * n - 1) as f32, m * n)
        .into_shape_with_order((m, n))
        .unwrap()
}

#[cfg(feature = "std")]
fn mean_axis0(bench: &mut Bencher)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    bench.iter(|| a.mean_axis(Axis(0)));
}

#[cfg(feature = "std")]
fn mean_axis1(bench: &mut Bencher)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    bench.iter(|| a.mean_axis(Axis(1)));
}

#[cfg(feature = "std")]
fn sum_axis0(bench: &mut Bencher)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    bench.iter(|| a.sum_axis(Axis(0)));
}

#[cfg(feature = "std")]
fn sum_axis1(bench: &mut Bencher)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    bench.iter(|| a.sum_axis(Axis(1)));
}

fn into_dimensionality_ix1_ok(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(Ix1(10));
    let a = a.view();
    bench.iter(|| a.into_dimensionality::<Ix1>());
}

fn into_dimensionality_ix3_ok(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(Ix3(10, 10, 10));
    let a = a.view();
    bench.iter(|| a.into_dimensionality::<Ix3>());
}

fn into_dimensionality_ix3_err(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(Ix3(10, 10, 10));
    let a = a.view();
    bench.iter(|| a.into_dimensionality::<Ix2>());
}

fn into_dimensionality_dyn_to_ix3(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(IxDyn(&[10, 10, 10]));
    let a = a.view();
    bench.iter(|| a.clone().into_dimensionality::<Ix3>());
}

fn into_dimensionality_dyn_to_dyn(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(IxDyn(&[10, 10, 10]));
    let a = a.view();
    bench.iter(|| a.clone().into_dimensionality::<IxDyn>());
}

fn into_dyn_ix3(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(Ix3(10, 10, 10));
    let a = a.view();
    bench.iter(|| a.into_dyn());
}

fn into_dyn_ix5(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(Ix5(2, 2, 2, 2, 2));
    let a = a.view();
    bench.iter(|| a.into_dyn());
}

fn into_dyn_dyn(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros(IxDyn(&[10, 10, 10]));
    let a = a.view();
    bench.iter(|| a.clone().into_dyn());
}

fn broadcast_same_dim(bench: &mut Bencher)
{
    let s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s = Array4::from_shape_vec((2, 2, 3, 2), s.to_vec()).unwrap();
    let a = s.slice(s![.., ..;-1, ..;2, ..]);
    let b = s.slice(s![.., .., ..;2, ..]);
    bench.iter(|| &a + &b);
}

fn broadcast_one_side(bench: &mut Bencher)
{
    let s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s2 = [1, 2, 3, 4, 5, 6];
    let a = Array4::from_shape_vec((4, 1, 3, 2), s.to_vec()).unwrap();
    let b = Array3::from_shape_vec((1, 3, 2), s2.to_vec()).unwrap();
    bench.iter(|| &a + &b);
}

benchmark_group!(
    benches,
    iter_sum_1d_regular,
    iter_sum_1d_raw,
    iter_sum_2d_regular,
    iter_sum_2d_by_row,
    iter_sum_2d_raw,
    iter_sum_2d_cutout,
    iter_sum_2d_cutout_by_row,
    iter_sum_2d_cutout_outer_iter,
    iter_sum_2d_transpose_regular,
    iter_sum_2d_transpose_by_row,
    sum_2d_regular,
    sum_2d_cutout,
    sum_2d_float,
    sum_2d_float_cutout,
    sum_2d_float_t_cutout,
    fold_sum_i32_2d_regular,
    fold_sum_i32_2d_cutout,
    fold_sum_i32_2d_stride,
    fold_sum_i32_2d_transpose,
    fold_sum_i32_2d_cutout_transpose,
    add_2d_regular,
    add_2d_zip,
    add_2d_alloc_plus,
    add_2d_alloc_zip_uninit,
    add_2d_alloc_zip_collect,
    vec_string_collect,
    array_string_collect,
    vec_f64_collect,
    array_f64_collect,
    add_2d_assign_ops,
    add_2d_cutout,
    add_2d_zip_cutout,
    add_2d_cutouts_by_4,
    add_2d_cutouts_by_16,
    add_2d_cutouts_by_32,
    add_2d_broadcast_1_to_2,
    add_2d_broadcast_0_to_2,
    scalar_toowned,
    scalar_add_1,
    scalar_add_2,
    scalar_add_strided_1,
    scalar_add_strided_2,
    scalar_sub_1,
    scalar_sub_2,
    add_2d_0_to_2_iadd_scalar,
    add_2d_strided,
    add_2d_regular_dyn,
    add_2d_strided_dyn,
    add_2d_zip_strided,
    add_2d_one_transposed,
    add_2d_zip_one_transposed,
    add_2d_both_transposed,
    add_2d_zip_both_transposed,
    add_2d_f32_regular,
    add_3d_strided,
    add_3d_strided_dyn,
    add_1d_regular,
    add_1d_strided,
    iadd_scalar_2d_regular,
    iadd_scalar_2d_strided,
    iadd_scalar_2d_regular_dyn,
    iadd_scalar_2d_strided_dyn,
    scaled_add_2d_f32_regular,
    assign_scalar_2d_corder,
    assign_scalar_2d_cutout,
    assign_scalar_2d_forder,
    assign_zero_2d_corder,
    assign_zero_2d_cutout,
    assign_zero_2d_forder,
    bench_iter_diag,
    bench_row_iter,
    bench_col_iter,
    create_iter_4d,
    bench_to_owned_n,
    bench_to_owned_t,
    bench_to_owned_strided,
    equality_i32,
    equality_f32,
    equality_f32_mixorder,
    dot_f32_16,
    dot_f32_20,
    dot_f32_32,
    dot_f32_256,
    dot_f32_1024,
    dot_f32_10e6,
    dot_extended,
    mean_axis0,
    mean_axis1,
    sum_axis0,
    sum_axis1,
    into_dimensionality_ix1_ok,
    into_dimensionality_ix3_ok,
    into_dimensionality_ix3_err,
    into_dimensionality_dyn_to_ix3,
    into_dimensionality_dyn_to_dyn,
    into_dyn_ix3,
    into_dyn_ix5,
    into_dyn_dyn,
    broadcast_same_dim,
    broadcast_one_side,
);
benchmark_main!(benches, mat_mul_f32, mat_mul_f64, mat_mul_i32);

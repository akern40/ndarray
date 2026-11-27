//! Benchmarks for [`SliceView`] performance, mainly iteration.

use divan::Bencher;
use nd_core::SliceView;
use nd_layout::{ShapeN, StridedLayout};
use ndarray::{Array2, ArrayView2, ShapeBuilder};

const ROWS: usize = 128;
const COLS: usize = 128;
const ROW_STRIDE: usize = COLS * 2;
const STRIDED_LEN: usize = (ROWS - 1) * ROW_STRIDE + COLS;

fn contiguous_data() -> Vec<u64> {
    (0..ROWS * COLS).map(|value| value as u64).collect()
}

fn strided_data() -> Vec<u64> {
    (0..STRIDED_LEN).map(|value| value as u64).collect()
}

fn contiguous_layout() -> StridedLayout<2> {
    let Some(layout) = StridedLayout::try_new(ShapeN::new([ROWS, COLS]), [COLS.cast_signed(), 1])
    else {
        unreachable!()
    };
    layout
}

fn row_strided_data_sum(data: &[u64]) -> u64 {
    data.iter()
        .enumerate()
        .filter(|(position, _)| position % ROW_STRIDE < COLS)
        .map(|(_, value)| *value)
        .sum()
}

fn reversed_axes_values(data: &[u64]) -> Vec<u64> {
    let (rows, remainder) = data.as_chunks::<COLS>();
    assert_eq!(remainder.len(), 0);

    let mut values = Vec::new();
    for column in 0..COLS {
        for row in rows {
            let Some(value) = row.get(column) else {
                unreachable!()
            };
            values.push(*value);
        }
    }
    values
}

fn reversed_row_strided_values(data: &[u64]) -> Vec<u64> {
    let mut values = Vec::new();
    for column in 0..COLS {
        for row in 0..ROWS {
            let Some(value) = data.get(row * ROW_STRIDE + column) else {
                unreachable!()
            };
            values.push(*value);
        }
    }
    values
}

fn row_strided_layout() -> StridedLayout<2> {
    let Some(layout) =
        StridedLayout::try_new(ShapeN::new([ROWS, COLS]), [ROW_STRIDE.cast_signed(), 1])
    else {
        unreachable!()
    };
    layout
}

#[divan::bench]
fn slice_view_contiguous_iter_sum(bencher: Bencher<'_, '_>) {
    let data = contiguous_data();
    let Some(view) = SliceView::try_new(&data, contiguous_layout(), 0) else {
        unreachable!()
    };
    assert_eq!(view.iter().copied().sum::<u64>(), data.iter().sum::<u64>());
    let view = divan::black_box(view);

    bencher.bench_local(|| {
        divan::black_box(view.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn ndarray_contiguous_iter_sum(bencher: Bencher<'_, '_>) {
    let data = contiguous_data();
    let expected_sum = data.iter().sum::<u64>();
    let Ok(array) = Array2::from_shape_vec((ROWS, COLS), data) else {
        unreachable!()
    };
    assert_eq!(array.iter().copied().sum::<u64>(), expected_sum);
    let array = divan::black_box(array);

    bencher.bench_local(|| {
        divan::black_box(array.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn rust_slice_contiguous_iter_sum(bencher: Bencher<'_, '_>) {
    let data = divan::black_box(contiguous_data());

    bencher.bench_local(|| {
        divan::black_box(data.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn slice_view_row_strided_iter_sum(bencher: Bencher<'_, '_>) {
    let data = strided_data();
    let Some(view) = SliceView::try_new(&data, row_strided_layout(), 0) else {
        unreachable!()
    };
    assert_eq!(
        view.iter().copied().sum::<u64>(),
        row_strided_data_sum(&data)
    );
    let view = divan::black_box(view);

    bencher.bench_local(|| {
        divan::black_box(view.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn ndarray_row_strided_iter_sum(bencher: Bencher<'_, '_>) {
    let data = strided_data();
    let Ok(view) = ArrayView2::from_shape((ROWS, COLS).strides((ROW_STRIDE, 1)), &data) else {
        unreachable!()
    };
    assert_eq!(
        view.iter().copied().sum::<u64>(),
        row_strided_data_sum(&data)
    );
    let view = divan::black_box(view);

    bencher.bench_local(|| {
        divan::black_box(view.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn slice_view_reversed_axes_iter_sum(bencher: Bencher<'_, '_>) {
    let data = contiguous_data();
    let Some(view) = SliceView::try_new(&data, contiguous_layout(), 0) else {
        unreachable!()
    };
    let reversed = view.reverse_axes();
    let expected = reversed_axes_values(&data);
    assert_eq!(reversed.iter().copied().collect::<Vec<_>>(), expected);
    let reversed = divan::black_box(reversed);

    bencher.bench_local(|| {
        divan::black_box(reversed.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn ndarray_reversed_axes_iter_sum(bencher: Bencher<'_, '_>) {
    let data = contiguous_data();
    let expected = reversed_axes_values(&data);
    let Ok(array) = Array2::from_shape_vec((ROWS, COLS), data) else {
        unreachable!()
    };
    let reversed = array.view().reversed_axes();
    assert_eq!(reversed.iter().copied().collect::<Vec<_>>(), expected);
    let reversed = divan::black_box(reversed);

    bencher.bench_local(|| {
        divan::black_box(reversed.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn slice_view_reversed_row_strided_iter_sum(bencher: Bencher<'_, '_>) {
    let data = strided_data();
    let Some(view) = SliceView::try_new(&data, row_strided_layout(), 0) else {
        unreachable!()
    };
    let reversed = view.reverse_axes();
    let expected = reversed_row_strided_values(&data);
    assert_eq!(reversed.iter().copied().collect::<Vec<_>>(), expected);
    let reversed = divan::black_box(reversed);

    bencher.bench_local(|| {
        divan::black_box(reversed.iter().copied().sum::<u64>());
    });
}

#[divan::bench]
fn ndarray_reversed_row_strided_iter_sum(bencher: Bencher<'_, '_>) {
    let data = strided_data();
    let expected = reversed_row_strided_values(&data);
    let Ok(strided) = ArrayView2::from_shape((ROWS, COLS).strides((ROW_STRIDE, 1)), &data) else {
        unreachable!()
    };
    let reversed = strided.reversed_axes();
    assert_eq!(reversed.iter().copied().collect::<Vec<_>>(), expected);
    let reversed = divan::black_box(reversed);

    bencher.bench_local(|| {
        divan::black_box(reversed.iter().copied().sum::<u64>());
    });
}

fn main() {
    divan::main();
}

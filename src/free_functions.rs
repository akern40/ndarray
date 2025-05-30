// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use meshgrid_impl::Meshgrid;
#[allow(unused_imports)]
use std::compile_error;
use std::mem::{forget, size_of};
use std::ptr::NonNull;

use crate::{dimension, ArcArray1, ArcArray2};
use crate::{imp_prelude::*, LayoutRef};

/// Create an **[`Array`]** with one, two, three, four, five, or six dimensions.
///
/// ```
/// use ndarray::array;
/// let a1 = array![1, 2, 3, 4];
///
/// let a2 = array![[1, 2],
///                 [3, 4]];
///
/// let a3 = array![[[1, 2], [3, 4]],
///                 [[5, 6], [7, 8]]];
///
/// let a4 = array![[[[1, 2, 3, 4]]]];
///
/// let a5 = array![[[[[1, 2, 3, 4, 5]]]]];
///
/// let a6 = array![[[[[[1, 2, 3, 4, 5, 6]]]]]];
///
/// assert_eq!(a1.shape(), &[4]);
/// assert_eq!(a2.shape(), &[2, 2]);
/// assert_eq!(a3.shape(), &[2, 2, 2]);
/// assert_eq!(a4.shape(), &[1, 1, 1, 4]);
/// assert_eq!(a5.shape(), &[1, 1, 1, 1, 5]);
/// assert_eq!(a6.shape(), &[1, 1, 1, 1, 1, 6]);
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
/// If running with `no_std`, this may require that you `use alloc::vec`
/// before being able to use the `array!` macro.
///
/// Use `array![...].into_shared()` to create an `ArcArray`.
///
/// Attempts to crate 7D+ arrays with this macro will lead to
/// a compiler error, since the difference between a 7D array
/// of i32 and a 6D array of `[i32; 3]` is ambiguous. Higher-dim
/// arrays can be created with [`ArrayD`].
///
/// ```compile_fail
/// use ndarray::array;
/// let a7 = array![[[[[[[1, 2, 3]]]]]]];
/// // error: Arrays of 7 dimensions or more (or ndarrays of Rust arrays) cannot be constructed with the array! macro.
/// ```
#[macro_export]
macro_rules! array {
    ($([$([$([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*) => {{
        compile_error!("Arrays of 7 dimensions or more (or ndarrays of Rust arrays) cannot be constructed with the array! macro.");
    }};
    ($([$([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array6::from(vec![$([$([$([$([$([$($x,)*],)*],)*],)*],)*],)*])
    }};
    ($([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array5::from(vec![$([$([$([$([$($x,)*],)*],)*],)*],)*])
    }};
    ($([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array4::from(vec![$([$([$([$($x,)*],)*],)*],)*])
    }};
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array3::from(vec![$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Array2::from(vec![$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Array::from(vec![$($x,)*])
    }};
}

/// Create a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> Array0<A>
{
    unsafe { ArrayBase::from_shape_vec_unchecked((), vec![x]) }
}

/// Create a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> Array1<A>
{
    ArrayBase::from(xs.to_vec())
}

/// Create a one-dimensional array with elements from `xs`.
pub fn rcarr1<A: Clone>(xs: &[A]) -> ArcArray1<A>
{
    arr1(xs).into_shared()
}

/// Create a zero-dimensional array view borrowing `x`.
pub const fn aview0<A>(x: &A) -> ArrayView0<'_, A>
{
    ArrayBase {
        data: ViewRepr::new(),
        layout: LayoutRef {
            // Safe because references are always non-null.
            ptr: unsafe { NonNull::new_unchecked(x as *const A as *mut A) },
            dim: Ix0(),
            strides: Ix0(),
        },
    }
}

/// Create a one-dimensional array view with elements borrowing `xs`.
///
/// **Panics** if the length of the slice overflows `isize`. (This can only
/// occur if `A` is zero-sized, because slices cannot contain more than
/// `isize::MAX` number of bytes.)
///
/// ```
/// use ndarray::{aview1, ArrayView1};
///
/// let data = [1.0; 1024];
///
/// // Create a 2D array view from borrowed data
/// let a2d = aview1(&data).into_shape_with_order((32, 32)).unwrap();
///
/// assert_eq!(a2d.sum(), 1024.0);
///
/// // Create a const 1D array view
/// const C: ArrayView1<'static, f64> = aview1(&[1., 2., 3.]);
///
/// assert_eq!(C.sum(), 6.);
/// ```
pub const fn aview1<A>(xs: &[A]) -> ArrayView1<'_, A>
{
    if size_of::<A>() == 0 {
        assert!(
            xs.len() <= isize::MAX as usize,
            "Slice length must fit in `isize`.",
        );
    }
    ArrayBase {
        data: ViewRepr::new(),
        layout: LayoutRef {
            // Safe because references are always non-null.
            ptr: unsafe { NonNull::new_unchecked(xs.as_ptr() as *mut A) },
            dim: Ix1(xs.len()),
            strides: Ix1(1),
        },
    }
}

/// Create a two-dimensional array view with elements borrowing `xs`.
///
/// **Panics** if the product of non-zero axis lengths overflows `isize` (This
/// can only occur if A is zero-sized or if `N` is zero, because slices cannot
/// contain more than `isize::MAX` number of bytes).
///
/// ```
/// use ndarray::{aview2, ArrayView2};
///
/// let data = vec![[1., 2., 3.], [4., 5., 6.]];
///
/// let view = aview2(&data);
/// assert_eq!(view.sum(), 21.);
///
/// // Create a const 2D array view
/// const C: ArrayView2<'static, f64> = aview2(&[[1., 2., 3.], [4., 5., 6.]]);
/// assert_eq!(C.sum(), 21.);
/// ```
pub const fn aview2<A, const N: usize>(xs: &[[A; N]]) -> ArrayView2<'_, A>
{
    let cols = N;
    let rows = xs.len();
    if size_of::<A>() == 0 {
        if let Some(n_elems) = rows.checked_mul(cols) {
            assert!(
                rows <= isize::MAX as usize
                    && cols <= isize::MAX as usize
                    && n_elems <= isize::MAX as usize,
                "Product of non-zero axis lengths must not overflow isize.",
            );
        } else {
            panic!("Overflow in number of elements.");
        }
    } else if N == 0 {
        assert!(
            rows <= isize::MAX as usize,
            "Product of non-zero axis lengths must not overflow isize.",
        );
    }
    // Safe because references are always non-null.
    let ptr = unsafe { NonNull::new_unchecked(xs.as_ptr() as *mut A) };
    let dim = Ix2(rows, cols);
    let strides = if rows == 0 || cols == 0 {
        Ix2(0, 0)
    } else {
        Ix2(cols, 1)
    };
    ArrayBase {
        data: ViewRepr::new(),
        layout: LayoutRef { ptr, dim, strides },
    }
}

/// Create a one-dimensional read-write array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::{aview_mut1, s};
/// // Create an array view over some data, then slice it and modify it.
/// let mut data = [0; 1024];
/// {
///     let mut a = aview_mut1(&mut data).into_shape_with_order((32, 32)).unwrap();
///     a.slice_mut(s![.., ..;3]).fill(5);
/// }
/// assert_eq!(&data[..10], [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]);
/// ```
pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut1<'_, A>
{
    ArrayViewMut::from(xs)
}

/// Create a two-dimensional read-write array view with elements borrowing `xs`.
///
/// **Panics** if the product of non-zero axis lengths overflows `isize` (This can only occur if A
/// is zero-sized because slices cannot contain more than `isize::MAX` number of bytes).
///
/// # Example
///
/// ```
/// use ndarray::aview_mut2;
///
/// // The inner (nested) and outer arrays can be of any length.
/// let mut data = [[0.; 2]; 128];
/// {
///     // Make a 128 x 2 mut array view then turn it into 2 x 128
///     let mut a = aview_mut2(&mut data).reversed_axes();
///     // Make the first row ones and second row minus ones.
///     a.row_mut(0).fill(1.);
///     a.row_mut(1).fill(-1.);
/// }
/// // look at the start of the result
/// assert_eq!(&data[..3], [[1., -1.], [1., -1.], [1., -1.]]);
/// ```
pub fn aview_mut2<A, const N: usize>(xs: &mut [[A; N]]) -> ArrayViewMut2<'_, A>
{
    ArrayViewMut2::from(xs)
}

/// Create a two-dimensional array with elements from `xs`.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1, 2, 3],
///                [4, 5, 6]]);
/// assert!(
///     a.shape() == [2, 3]
/// );
/// ```
pub fn arr2<A: Clone, const N: usize>(xs: &[[A; N]]) -> Array2<A>
{
    Array2::from(xs.to_vec())
}

macro_rules! impl_from_nested_vec {
    ($arr_type:ty, $ix_type:tt, $($n:ident),+) => {
        impl<A, $(const $n: usize),+> From<Vec<$arr_type>> for Array<A, $ix_type>
        {
            fn from(mut xs: Vec<$arr_type>) -> Self
            {
                let dim = $ix_type(xs.len(), $($n),+);
                let ptr = xs.as_mut_ptr();
                let cap = xs.capacity();
                let expand_len = dimension::size_of_shape_checked(&dim)
                    .expect("Product of non-zero axis lengths must not overflow isize.");
                forget(xs);
                unsafe {
                    let v = if size_of::<A>() == 0 {
                        Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
                    } else if $($n == 0 ||)+ false {
                        Vec::new()
                    } else {
                        let expand_cap = cap $(* $n)+;
                        Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
                    };
                    ArrayBase::from_shape_vec_unchecked(dim, v)
                }
            }
        }
    };
}

impl_from_nested_vec!([A; N], Ix2, N);
impl_from_nested_vec!([[A; M]; N], Ix3, N, M);
impl_from_nested_vec!([[[A; L]; M]; N], Ix4, N, M, L);
impl_from_nested_vec!([[[[A; K]; L]; M]; N], Ix5, N, M, L, K);
impl_from_nested_vec!([[[[[A; J]; K]; L]; M]; N], Ix6, N, M, L, K, J);

/// Create a two-dimensional array with elements from `xs`.
///
pub fn rcarr2<A: Clone, const N: usize>(xs: &[[A; N]]) -> ArcArray2<A>
{
    arr2(xs).into_shared()
}

/// Create a three-dimensional array with elements from `xs`.
///
/// **Panics** if the slices are not all of the same length.
///
/// ```
/// use ndarray::arr3;
///
/// let a = arr3(&[[[1, 2],
///                 [3, 4]],
///                [[5, 6],
///                 [7, 8]],
///                [[9, 0],
///                 [1, 2]]]);
/// assert!(
///     a.shape() == [3, 2, 2]
/// );
/// ```
pub fn arr3<A: Clone, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> Array3<A>
{
    Array3::from(xs.to_vec())
}

/// Create a three-dimensional array with elements from `xs`.
pub fn rcarr3<A: Clone, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> ArcArray<A, Ix3>
{
    arr3(xs).into_shared()
}

/// The indexing order for [`meshgrid`]; see there for more details.
///
/// Controls whether the first argument to `meshgrid` will fill the rows or columns of the outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshIndex
{
    /// Cartesian indexing.
    ///
    /// The first argument of `meshgrid` will repeat over the columns of the output.
    ///
    /// Note: this is the default in `numpy`.
    XY,
    /// Matrix indexing.
    ///
    /// The first argument of `meshgrid` will repeat over the rows of the output.
    IJ,
}

mod meshgrid_impl
{
    use super::MeshIndex;
    use crate::extension::nonnull::nonnull_debug_checked_from_ptr;
    use crate::{
        ArrayBase,
        ArrayRef1,
        ArrayView,
        ArrayView2,
        ArrayView3,
        ArrayView4,
        ArrayView5,
        ArrayView6,
        Axis,
        Data,
        Dim,
        IntoDimension,
        Ix1,
        LayoutRef1,
    };

    /// Construct the correct strides for the `idx`-th entry into meshgrid
    fn construct_strides<A, const N: usize>(
        arr: &LayoutRef1<A>, idx: usize, indexing: MeshIndex,
    ) -> <[usize; N] as IntoDimension>::Dim
    where [usize; N]: IntoDimension
    {
        let mut ret = [0; N];
        if idx < 2 && indexing == MeshIndex::XY {
            ret[1 - idx] = arr.stride_of(Axis(0)) as usize;
        } else {
            ret[idx] = arr.stride_of(Axis(0)) as usize;
        }
        Dim(ret)
    }

    /// Construct the correct shape for the `idx`-th entry into meshgrid
    fn construct_shape<A, const N: usize>(
        arrays: [&LayoutRef1<A>; N], indexing: MeshIndex,
    ) -> <[usize; N] as IntoDimension>::Dim
    where [usize; N]: IntoDimension
    {
        let mut ret = arrays.map(|a| a.len());
        if indexing == MeshIndex::XY {
            ret.swap(0, 1);
        }
        Dim(ret)
    }

    /// A trait to encapsulate static dispatch for [`meshgrid`](super::meshgrid); see there for more details.
    ///
    /// The inputs should always be some sort of 1D array.
    /// The outputs should always be ND arrays where N is the number of inputs.
    ///
    /// Where possible, this trait tries to return array views rather than allocating additional memory.
    pub trait Meshgrid
    {
        type Output;

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output;
    }

    macro_rules! meshgrid_body {
        ($count:literal, $indexing:expr, $(($arr:expr, $idx:literal)),+) => {
        {
            let shape = construct_shape([$($arr),+], $indexing);
            (
                $({
                    let strides = construct_strides::<_, $count>($arr, $idx, $indexing);
                    unsafe { ArrayView::new(nonnull_debug_checked_from_ptr($arr.as_ptr() as *mut A), shape, strides) }
                }),+
            )
        }
    };
    }

    impl<'a, 'b, A> Meshgrid for (&'a ArrayRef1<A>, &'b ArrayRef1<A>)
    {
        type Output = (ArrayView2<'a, A>, ArrayView2<'b, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            meshgrid_body!(2, indexing, (arrays.0, 0), (arrays.1, 1))
        }
    }

    impl<'a, 'b, S1, S2, A: 'b + 'a> Meshgrid for (&'a ArrayBase<S1, Ix1>, &'b ArrayBase<S2, Ix1>)
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A>,
    {
        type Output = (ArrayView2<'a, A>, ArrayView2<'b, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            Meshgrid::meshgrid((&**arrays.0, &**arrays.1), indexing)
        }
    }

    impl<'a, 'b, 'c, A> Meshgrid for (&'a ArrayRef1<A>, &'b ArrayRef1<A>, &'c ArrayRef1<A>)
    {
        type Output = (ArrayView3<'a, A>, ArrayView3<'b, A>, ArrayView3<'c, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            meshgrid_body!(3, indexing, (arrays.0, 0), (arrays.1, 1), (arrays.2, 2))
        }
    }

    impl<'a, 'b, 'c, S1, S2, S3, A: 'b + 'a + 'c> Meshgrid
        for (&'a ArrayBase<S1, Ix1>, &'b ArrayBase<S2, Ix1>, &'c ArrayBase<S3, Ix1>)
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A>,
        S3: Data<Elem = A>,
    {
        type Output = (ArrayView3<'a, A>, ArrayView3<'b, A>, ArrayView3<'c, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            Meshgrid::meshgrid((&**arrays.0, &**arrays.1, &**arrays.2), indexing)
        }
    }

    impl<'a, 'b, 'c, 'd, A> Meshgrid for (&'a ArrayRef1<A>, &'b ArrayRef1<A>, &'c ArrayRef1<A>, &'d ArrayRef1<A>)
    {
        type Output = (ArrayView4<'a, A>, ArrayView4<'b, A>, ArrayView4<'c, A>, ArrayView4<'d, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            meshgrid_body!(4, indexing, (arrays.0, 0), (arrays.1, 1), (arrays.2, 2), (arrays.3, 3))
        }
    }

    impl<'a, 'b, 'c, 'd, S1, S2, S3, S4, A: 'a + 'b + 'c + 'd> Meshgrid
        for (&'a ArrayBase<S1, Ix1>, &'b ArrayBase<S2, Ix1>, &'c ArrayBase<S3, Ix1>, &'d ArrayBase<S4, Ix1>)
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A>,
        S3: Data<Elem = A>,
        S4: Data<Elem = A>,
    {
        type Output = (ArrayView4<'a, A>, ArrayView4<'b, A>, ArrayView4<'c, A>, ArrayView4<'d, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            Meshgrid::meshgrid((&**arrays.0, &**arrays.1, &**arrays.2, &**arrays.3), indexing)
        }
    }

    impl<'a, 'b, 'c, 'd, 'e, A> Meshgrid
        for (&'a ArrayRef1<A>, &'b ArrayRef1<A>, &'c ArrayRef1<A>, &'d ArrayRef1<A>, &'e ArrayRef1<A>)
    {
        type Output = (ArrayView5<'a, A>, ArrayView5<'b, A>, ArrayView5<'c, A>, ArrayView5<'d, A>, ArrayView5<'e, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            meshgrid_body!(5, indexing, (arrays.0, 0), (arrays.1, 1), (arrays.2, 2), (arrays.3, 3), (arrays.4, 4))
        }
    }

    impl<'a, 'b, 'c, 'd, 'e, S1, S2, S3, S4, S5, A: 'a + 'b + 'c + 'd + 'e> Meshgrid
        for (
            &'a ArrayBase<S1, Ix1>,
            &'b ArrayBase<S2, Ix1>,
            &'c ArrayBase<S3, Ix1>,
            &'d ArrayBase<S4, Ix1>,
            &'e ArrayBase<S5, Ix1>,
        )
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A>,
        S3: Data<Elem = A>,
        S4: Data<Elem = A>,
        S5: Data<Elem = A>,
    {
        type Output = (ArrayView5<'a, A>, ArrayView5<'b, A>, ArrayView5<'c, A>, ArrayView5<'d, A>, ArrayView5<'e, A>);

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            Meshgrid::meshgrid((&**arrays.0, &**arrays.1, &**arrays.2, &**arrays.3, &**arrays.4), indexing)
        }
    }

    impl<'a, 'b, 'c, 'd, 'e, 'f, A> Meshgrid
        for (
            &'a ArrayRef1<A>,
            &'b ArrayRef1<A>,
            &'c ArrayRef1<A>,
            &'d ArrayRef1<A>,
            &'e ArrayRef1<A>,
            &'f ArrayRef1<A>,
        )
    {
        type Output = (
            ArrayView6<'a, A>,
            ArrayView6<'b, A>,
            ArrayView6<'c, A>,
            ArrayView6<'d, A>,
            ArrayView6<'e, A>,
            ArrayView6<'f, A>,
        );

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            meshgrid_body!(6, indexing, (arrays.0, 0), (arrays.1, 1), (arrays.2, 2), (arrays.3, 3), (arrays.4, 4), (arrays.5, 5))
        }
    }

    impl<'a, 'b, 'c, 'd, 'e, 'f, S1, S2, S3, S4, S5, S6, A: 'a + 'b + 'c + 'd + 'e + 'f> Meshgrid
        for (
            &'a ArrayBase<S1, Ix1>,
            &'b ArrayBase<S2, Ix1>,
            &'c ArrayBase<S3, Ix1>,
            &'d ArrayBase<S4, Ix1>,
            &'e ArrayBase<S5, Ix1>,
            &'f ArrayBase<S6, Ix1>,
        )
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A>,
        S3: Data<Elem = A>,
        S4: Data<Elem = A>,
        S5: Data<Elem = A>,
        S6: Data<Elem = A>,
    {
        type Output = (
            ArrayView6<'a, A>,
            ArrayView6<'b, A>,
            ArrayView6<'c, A>,
            ArrayView6<'d, A>,
            ArrayView6<'e, A>,
            ArrayView6<'f, A>,
        );

        fn meshgrid(arrays: Self, indexing: MeshIndex) -> Self::Output
        {
            Meshgrid::meshgrid((&**arrays.0, &**arrays.1, &**arrays.2, &**arrays.3, &**arrays.4, &**arrays.5), indexing)
        }
    }
}

/// Create coordinate matrices from coordinate vectors.
///
/// Given an N-tuple of 1D coordinate vectors, return an N-tuple of ND coordinate arrays.
/// This is particularly useful for computing the outputs of functions with N arguments over
/// regularly spaced grids.
///
/// The `indexing` argument can be controlled by [`MeshIndex`] to support both Cartesian and
/// matrix indexing. In the two-dimensional case, inputs of length `N` and `M` will create
/// output arrays of size `(M, N)` when using [`MeshIndex::XY`] and size `(N, M)` when using
/// [`MeshIndex::IJ`].
///
/// # Example
/// ```
/// use ndarray::{array, meshgrid, MeshIndex};
///
/// let arr1 = array![1, 2];
/// let arr2 = array![3, 4];
/// let arr3 = array![5, 6];
///
/// // Cartesian indexing
/// let (res1, res2) = meshgrid((&arr1, &arr2), MeshIndex::XY);
/// assert_eq!(res1, array![
///     [1, 2],
///     [1, 2],
/// ]);
/// assert_eq!(res2, array![
///     [3, 3],
///     [4, 4],
/// ]);
///
/// // Matrix indexing
/// let (res1, res2) = meshgrid((&arr1, &arr2), MeshIndex::IJ);
/// assert_eq!(res1, array![
///     [1, 1],
///     [2, 2],
/// ]);
/// assert_eq!(res2, array![
///     [3, 4],
///     [3, 4],
/// ]);
///
/// let (_, _, res3) = meshgrid((&arr1, &arr2, &arr3), MeshIndex::XY);
/// assert_eq!(res3, array![
///     [[5, 6],
///      [5, 6]],
///     [[5, 6],
///      [5, 6]],
/// ]);
/// ```
pub fn meshgrid<T: Meshgrid>(arrays: T, indexing: MeshIndex) -> T::Output
{
    Meshgrid::meshgrid(arrays, indexing)
}

#[cfg(test)]
mod tests
{
    use super::s;
    use crate::{meshgrid, Axis, MeshIndex};
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_meshgrid2()
    {
        let x = array![1, 2, 3];
        let y = array![4, 5, 6, 7];
        let (xx, yy) = meshgrid((&x, &y), MeshIndex::XY);
        assert_eq!(xx, array![[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]);
        assert_eq!(yy, array![[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]);

        let (xx, yy) = meshgrid((&x, &y), MeshIndex::IJ);
        assert_eq!(xx, array![[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
        assert_eq!(yy, array![[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]);
    }

    #[test]
    fn test_meshgrid3()
    {
        let x = array![1, 2, 3];
        let y = array![4, 5, 6, 7];
        let z = array![-1, -2];
        let (xx, yy, zz) = meshgrid((&x, &y, &z), MeshIndex::XY);
        assert_eq!(xx, array![
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
        ]);
        assert_eq!(yy, array![
            [[4, 4], [4, 4], [4, 4]],
            [[5, 5], [5, 5], [5, 5]],
            [[6, 6], [6, 6], [6, 6]],
            [[7, 7], [7, 7], [7, 7]],
        ]);
        assert_eq!(zz, array![
            [[-1, -2], [-1, -2], [-1, -2]],
            [[-1, -2], [-1, -2], [-1, -2]],
            [[-1, -2], [-1, -2], [-1, -2]],
            [[-1, -2], [-1, -2], [-1, -2]],
        ]);

        let (xx, yy, zz) = meshgrid((&x, &y, &z), MeshIndex::IJ);
        assert_eq!(xx, array![
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            [[2, 2], [2, 2], [2, 2], [2, 2]],
            [[3, 3], [3, 3], [3, 3], [3, 3]],
        ]);
        assert_eq!(yy, array![
            [[4, 4], [5, 5], [6, 6], [7, 7]],
            [[4, 4], [5, 5], [6, 6], [7, 7]],
            [[4, 4], [5, 5], [6, 6], [7, 7]],
        ]);
        assert_eq!(zz, array![
            [[-1, -2], [-1, -2], [-1, -2], [-1, -2]],
            [[-1, -2], [-1, -2], [-1, -2], [-1, -2]],
            [[-1, -2], [-1, -2], [-1, -2], [-1, -2]],
        ]);
    }

    #[test]
    fn test_meshgrid_from_offset()
    {
        let x = array![1, 2, 3];
        let x = x.slice(s![1..]);
        let y = array![4, 5, 6];
        let y = y.slice(s![1..]);
        let (xx, yy) = meshgrid((&x, &y), MeshIndex::XY);
        assert_eq!(xx, array![[2, 3], [2, 3]]);
        assert_eq!(yy, array![[5, 5], [6, 6]]);
    }

    #[test]
    fn test_meshgrid_neg_stride()
    {
        let x = array![1, 2, 3];
        let x = x.slice(s![..;-1]);
        assert!(x.stride_of(Axis(0)) < 0); // Setup for test
        let y = array![4, 5, 6];
        let (xx, yy) = meshgrid((&x, &y), MeshIndex::XY);
        assert_eq!(xx, array![[3, 2, 1], [3, 2, 1], [3, 2, 1]]);
        assert_eq!(yy, array![[4, 4, 4], [5, 5, 5], [6, 6, 6]]);
    }
}

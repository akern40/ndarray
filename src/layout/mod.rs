mod layoutfmt;
pub mod dimensionality;
mod dyn_repr;
mod n_repr;
pub mod shape;
pub mod strides;
pub use dyn_repr::{DShape, DStrides};
pub use n_repr::{NShape, NStrides};

use alloc::borrow::Cow;
use core::{any::type_name, fmt::Display, marker::PhantomData};
use dimensionality::{Dimensionality, NDim};
pub use shape::Shape;
pub use strides::Strides;

// Layout is a bitset used for internal layout description of
// arrays, producers and sets of producers.
// The type is public but users don't interact with it.
#[doc(hidden)]
/// Memory layout description
#[derive(Copy, Clone)]
pub struct LayoutBitset(u32);

impl LayoutBitset
{
    pub(crate) const CORDER: u32 = 0b01;
    pub(crate) const FORDER: u32 = 0b10;
    pub(crate) const CPREFER: u32 = 0b0100;
    pub(crate) const FPREFER: u32 = 0b1000;

    #[inline(always)]
    pub(crate) fn is(self, flag: u32) -> bool
    {
        self.0 & flag != 0
    }

    /// Return layout common to both inputs
    #[inline(always)]
    pub(crate) fn intersect(self, other: LayoutBitset) -> LayoutBitset
    {
        LayoutBitset(self.0 & other.0)
    }

    /// Return a layout that simultaneously "is" what both of the inputs are
    #[inline(always)]
    pub(crate) fn also(self, other: LayoutBitset) -> LayoutBitset
    {
        LayoutBitset(self.0 | other.0)
    }

    #[inline(always)]
    pub(crate) fn one_dimensional() -> LayoutBitset
    {
        LayoutBitset::c().also(LayoutBitset::f())
    }

    #[inline(always)]
    pub(crate) fn c() -> LayoutBitset
    {
        LayoutBitset(LayoutBitset::CORDER | LayoutBitset::CPREFER)
    }

    #[inline(always)]
    pub(crate) fn f() -> LayoutBitset
    {
        LayoutBitset(LayoutBitset::FORDER | LayoutBitset::FPREFER)
    }

    #[inline(always)]
    pub(crate) fn cpref() -> LayoutBitset
    {
        LayoutBitset(LayoutBitset::CPREFER)
    }

    #[inline(always)]
    pub(crate) fn fpref() -> LayoutBitset
    {
        LayoutBitset(LayoutBitset::FPREFER)
    }

    #[inline(always)]
    pub(crate) fn none() -> LayoutBitset
    {
        LayoutBitset(0)
    }

    /// A simple "score" method which scores positive for preferring C-order, negative for F-order
    /// Subject to change when we can describe other layouts
    #[inline]
    pub(crate) fn tendency(self) -> i32
    {
        (self.is(LayoutBitset::CORDER) as i32 - self.is(LayoutBitset::FORDER) as i32)
            + (self.is(LayoutBitset::CPREFER) as i32 - self.is(LayoutBitset::FPREFER) as i32)
    }
}

/// The error type for dealing with shapes and strides
#[derive(Debug, Clone, Copy)]
pub enum ShapeStrideError<S>
{
    /// Out of bounds; specifically, using an index that is larger than the dimensionality of the shape or strides `S`.
    OutOfBounds(PhantomData<S>, usize),
    /// The error when trying to mutate a shape or strides whose element is a hard-coded constant.
    FixedIndex(PhantomData<S>, usize),
    /// The error when trying to construct or mutate a shape or strides with the wrong dimensionality value.
    BadDimality(PhantomData<S>, usize),
}

impl<S: Strides> Display for ShapeStrideError<S>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
    {
        match self {
            ShapeStrideError::FixedIndex(_, idx) => write!(f, "Cannot index {} at {idx}", type_name::<S>()),
            ShapeStrideError::OutOfBounds(_, idx) =>
                write!(f, "Index {idx} is larger than the dimensionality of {}", type_name::<S>()),
            ShapeStrideError::BadDimality(_, dimality) => write!(f, "{} has a dimensionality of {}, which is incompatible with requested dimensionality of {dimality}", type_name::<S>(), type_name::<S::Dimality>()),
        }
    }
}

/// Trait that associates a dimensionality to a type
pub trait Dimensioned
{
    type Dimality: Dimensionality;
}

impl<T, const N: usize> Dimensioned for [T; N]
where NDim<N>: Dimensionality
{
    type Dimality = NDim<N>;
}

/// A trait capturing how an array is laid out in memory.
pub trait Layout: Dimensioned
{
    /// The type of shape that the array uses.
    ///
    /// Must implement [`Shape`] and have the same dimensionality.
    type Shape: Shape<Dimality = Self::Dimality>;

    /// The index type that this layout uses; e.g., `[usize; N]`.
    ///
    /// Must have the same dimensionality.
    type Index: Dimensioned<Dimality = Self::Dimality>;

    /// Get the shape of the layout.
    ///
    /// If the implementing type does not carry a shape directly,
    /// one should be constructed and passed as [`Cow::Owned`].
    fn shape(&self) -> Cow<'_, Self::Shape>;

    /// Index into this layout in "linear" fashion by moving across axes from left to right.
    fn index_linear_left(&self, idx: usize) -> isize;

    /// Index into this layout in "linear" fashion by moving across axes from right to left.
    fn index_linear_right(&self, idx: usize) -> isize;

    /// Index into this layout by traversing it in a memory order that is as efficient as possible.
    fn index_memory_order(&self, idx: usize) -> isize;

    /// Index into this layout with a multidimensional index.
    fn index(&self, idx: Self::Index) -> isize;

    fn first_index(&self) -> Option<Self::Index>;

    fn next_for(&self, index: Self::Index) -> Option<Self::Index>;

    // Shortcut methods, we could add more of these
    fn ndim(&self) -> usize
    {
        self.shape().ndim()
    }

    fn size(&self) -> usize
    {
        self.shape().size()
    }

    fn size_checked(&self) -> Option<usize>
    {
        self.shape().size_checked()
    }
}

pub trait Strided: Layout
{
    type Strides: Strides<Dimality = Self::Dimality>;

    fn strides(&self) -> Cow<'_, Self::Strides>;
}

pub struct NLayout<const N: usize>
{
    shape: NShape<N>,
    strides: NStrides<N>,
}

impl<const N: usize> Dimensioned for NLayout<N>
where NDim<N>: Dimensionality
{
    type Dimality = NDim<N>;
}

impl<const N: usize> Layout for NLayout<N>
where NDim<N>: Dimensionality
{
    type Shape = NShape<N>;

    type Index = [usize; N];

    fn shape(&self) -> Cow<'_, Self::Shape>
    {
        Cow::Borrowed(&self.shape)
    }

    fn index_linear_left(&self, idx: usize) -> isize
    {
        todo!()
    }

    fn index_linear_right(&self, idx: usize) -> isize
    {
        todo!()
    }

    fn index_memory_order(&self, idx: usize) -> isize
    {
        todo!()
    }

    fn index(&self, index: Self::Index) -> isize
    {
        let mut offset = 0isize;
        for idx in 0..N {
            offset += (index[idx] as isize) * self.strides[idx];
        }
        offset
    }

    fn first_index(&self) -> Option<Self::Index>
    {
        todo!()
    }

    fn next_for(&self, index: Self::Index) -> Option<Self::Index>
    {
        todo!()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::imp_prelude::*;
    use crate::NdProducer;

    type M = Array2<f32>;
    type M1 = Array1<f32>;
    type M0 = Array0<f32>;

    macro_rules! assert_layouts {
        ($mat:expr, $($layout:ident),*) => {{
            let layout = $mat.view().layout();
            $(
            assert!(layout.is(LayoutBitset::$layout),
                "Assertion failed: array {:?} is not layout {}",
                $mat,
                stringify!($layout));
            )*
        }};
    }

    macro_rules! assert_not_layouts {
        ($mat:expr, $($layout:ident),*) => {{
            let layout = $mat.view().layout();
            $(
            assert!(!layout.is(LayoutBitset::$layout),
                "Assertion failed: array {:?} show not have layout {}",
                $mat,
                stringify!($layout));
            )*
        }};
    }

    #[test]
    fn contig_layouts()
    {
        let a = M::zeros((5, 5));
        let b = M::zeros((5, 5).f());
        let ac = a.view().layout();
        let af = b.view().layout();
        assert!(ac.is(LayoutBitset::CORDER) && ac.is(LayoutBitset::CPREFER));
        assert!(!ac.is(LayoutBitset::FORDER) && !ac.is(LayoutBitset::FPREFER));
        assert!(!af.is(LayoutBitset::CORDER) && !af.is(LayoutBitset::CPREFER));
        assert!(af.is(LayoutBitset::FORDER) && af.is(LayoutBitset::FPREFER));
    }

    #[test]
    fn contig_cf_layouts()
    {
        let a = M::zeros((5, 1));
        let b = M::zeros((1, 5).f());
        assert_layouts!(a, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(b, CORDER, CPREFER, FORDER, FPREFER);

        let a = M1::zeros(5);
        let b = M1::zeros(5.f());
        assert_layouts!(a, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(b, CORDER, CPREFER, FORDER, FPREFER);

        let a = M0::zeros(());
        assert_layouts!(a, CORDER, CPREFER, FORDER, FPREFER);

        let a = M::zeros((5, 5));
        let b = M::zeros((5, 5).f());
        let arow = a.slice(s![..1, ..]);
        let bcol = b.slice(s![.., ..1]);
        assert_layouts!(arow, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(bcol, CORDER, CPREFER, FORDER, FPREFER);

        let acol = a.slice(s![.., ..1]);
        let brow = b.slice(s![..1, ..]);
        assert_not_layouts!(acol, CORDER, CPREFER, FORDER, FPREFER);
        assert_not_layouts!(brow, CORDER, CPREFER, FORDER, FPREFER);
    }

    #[test]
    fn stride_layouts()
    {
        let a = M::zeros((5, 5));

        {
            let v1 = a.slice(s![1.., ..]).layout();
            let v2 = a.slice(s![.., 1..]).layout();

            assert!(v1.is(LayoutBitset::CORDER) && v1.is(LayoutBitset::CPREFER));
            assert!(!v1.is(LayoutBitset::FORDER) && !v1.is(LayoutBitset::FPREFER));
            assert!(!v2.is(LayoutBitset::CORDER) && v2.is(LayoutBitset::CPREFER));
            assert!(!v2.is(LayoutBitset::FORDER) && !v2.is(LayoutBitset::FPREFER));
        }

        let b = M::zeros((5, 5).f());

        {
            let v1 = b.slice(s![1.., ..]).layout();
            let v2 = b.slice(s![.., 1..]).layout();

            assert!(!v1.is(LayoutBitset::CORDER) && !v1.is(LayoutBitset::CPREFER));
            assert!(!v1.is(LayoutBitset::FORDER) && v1.is(LayoutBitset::FPREFER));
            assert!(!v2.is(LayoutBitset::CORDER) && !v2.is(LayoutBitset::CPREFER));
            assert!(v2.is(LayoutBitset::FORDER) && v2.is(LayoutBitset::FPREFER));
        }
    }

    #[test]
    fn no_layouts()
    {
        let a = M::zeros((5, 5));
        let b = M::zeros((5, 5).f());

        // 2D row/column matrixes
        let arow = a.slice(s![0..1, ..]);
        let acol = a.slice(s![.., 0..1]);
        let brow = b.slice(s![0..1, ..]);
        let bcol = b.slice(s![.., 0..1]);
        assert_layouts!(arow, CORDER, FORDER);
        assert_not_layouts!(acol, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(bcol, CORDER, FORDER);
        assert_not_layouts!(brow, CORDER, CPREFER, FORDER, FPREFER);

        // 2D row/column matrixes - now made with insert axis
        for &axis in &[Axis(0), Axis(1)] {
            let arow = a.slice(s![0, ..]).insert_axis(axis);
            let acol = a.slice(s![.., 0]).insert_axis(axis);
            let brow = b.slice(s![0, ..]).insert_axis(axis);
            let bcol = b.slice(s![.., 0]).insert_axis(axis);
            assert_layouts!(arow, CORDER, FORDER);
            assert_not_layouts!(acol, CORDER, CPREFER, FORDER, FPREFER);
            assert_layouts!(bcol, CORDER, FORDER);
            assert_not_layouts!(brow, CORDER, CPREFER, FORDER, FPREFER);
        }
    }

    #[test]
    fn skip_layouts()
    {
        let a = M::zeros((5, 5));
        {
            let v1 = a.slice(s![..;2, ..]).layout();
            let v2 = a.slice(s![.., ..;2]).layout();

            assert!(!v1.is(LayoutBitset::CORDER) && v1.is(LayoutBitset::CPREFER));
            assert!(!v1.is(LayoutBitset::FORDER) && !v1.is(LayoutBitset::FPREFER));
            assert!(!v2.is(LayoutBitset::CORDER) && !v2.is(LayoutBitset::CPREFER));
            assert!(!v2.is(LayoutBitset::FORDER) && !v2.is(LayoutBitset::FPREFER));
        }

        let b = M::zeros((5, 5).f());
        {
            let v1 = b.slice(s![..;2, ..]).layout();
            let v2 = b.slice(s![.., ..;2]).layout();

            assert!(!v1.is(LayoutBitset::CORDER) && !v1.is(LayoutBitset::CPREFER));
            assert!(!v1.is(LayoutBitset::FORDER) && !v1.is(LayoutBitset::FPREFER));
            assert!(!v2.is(LayoutBitset::CORDER) && !v2.is(LayoutBitset::CPREFER));
            assert!(!v2.is(LayoutBitset::FORDER) && v2.is(LayoutBitset::FPREFER));
        }
    }
}

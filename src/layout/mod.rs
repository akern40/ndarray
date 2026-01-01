pub mod bitset;
pub mod checked;
mod bitsetfmt;
pub mod dimensionality;
mod dyn_repr;
mod n_repr;
pub mod shape;
pub mod strides;
pub mod strided_builder;

pub use dyn_repr::{DShape, DStrides};
pub use n_repr::{NShape, NStrides};
use num_traits::ToPrimitive;

use core::{any::type_name, error::Error, fmt::Debug, fmt::Display, marker::PhantomData};
use dimensionality::{Dimensionality, NDim};
pub use shape::Shape;
pub use strides::Strides;

use crate::layout::{dimensionality::DDyn, strides::DefaultC};

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
    /// The desired shape would represent an array with more elements than `usize::MAX`
    ShapeOverflow,
}

impl<S> ShapeStrideError<S>
{
    pub fn replace_type_with<T>(&self) -> ShapeStrideError<T>
    {
        match self {
            ShapeStrideError::OutOfBounds(_, u) => ShapeStrideError::OutOfBounds(PhantomData, *u),
            ShapeStrideError::FixedIndex(_, u) => ShapeStrideError::FixedIndex(PhantomData, *u),
            ShapeStrideError::BadDimality(_, u) => ShapeStrideError::BadDimality(PhantomData, *u),
            ShapeStrideError::ShapeOverflow => ShapeStrideError::ShapeOverflow,
        }
    }
}

impl<S: Dimensioned> Display for ShapeStrideError<S>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
    {
        match self {
            ShapeStrideError::FixedIndex(_, idx) => write!(f, "Cannot index {} at {idx}", type_name::<S>()),
            ShapeStrideError::OutOfBounds(_, idx) =>
                write!(f, "Index {idx} is larger than the dimensionality of {}", type_name::<S>()),
            ShapeStrideError::BadDimality(_, dimality) => write!(f, "{} has a dimensionality of {}, which is incompatible with requested dimensionality of {dimality}", type_name::<S>(), type_name::<S::Dimality>()),
            ShapeStrideError::ShapeOverflow => write!(f, "The desired shape would represent an array with more elements than `usize::MAX`")
        }
    }
}

impl<S: Debug + Dimensioned> Error for ShapeStrideError<S> {}

/// Trait that associates a dimensionality to a type
pub trait Dimensioned
{
    type Dimality: Dimensionality;

    fn ndim(&self) -> usize;
}

impl<T, const N: usize> Dimensioned for [T; N]
where NDim<N>: Dimensionality
{
    type Dimality = NDim<N>;

    fn ndim(&self) -> usize
    {
        N
    }
}

impl<T> Dimensioned for [T]
{
    type Dimality = DDyn;

    fn ndim(&self) -> usize
    {
        self.len()
    }
}

/// A trait capturing how an array is laid out in memory.
///
/// # Safety
/// Not all instances of a layout are valid. For example, layouts
/// with invalid shapes are themselves not valid; see [`Shape`] for
/// safety information. `Layout` instances must hold their own
/// invariants to be valid. In particular, the offset in bytes
/// between the minimum and maximum addresses of the array
/// cannot be greater than `isize::MAX`. This is true even if
/// the number of elements in the array is technically zero.
///
/// For example, if the layout is strided with a shape of `[2, 0, 3]`
/// and strides of `[3, 6, -1]`, the minimum offset is -2 and the maximum
/// offset is 3, with a total 5 elements between the minimum and maximum.
/// The offset in bytes is dependent on the type that the layout is representing.
///
/// To uphold these invariants, implementors of `Layout` guarantee that that
/// [`Layout::offset_range_checked`]:
/// 1. Only returns `Some(_)` if the offset in _number of elements_ fits in `isize`.
/// 2. When it returns `Some(_)`, it represents the _exact_ offset in `isize`.
///
/// Similar guarantees must be held for [`Layout::offset_range_bytes_checked`].
pub trait Layout: Dimensioned + Clone + Default
{
    /// The type of shape that the array uses.
    ///
    /// Must implement [`Shape`] and have the same dimensionality.
    type Shape: Shape<Dimality = Self::Dimality>;

    /// The index type that this layout uses; e.g., `[usize; N]`.
    ///
    /// Must have the same dimensionality.
    type Index: Dimensioned<Dimality = Self::Dimality> + ?Sized;

    /// Get the shape of the layout.
    ///
    /// If the implementing type does not carry a shape directly,
    /// one should be constructed and passed as [`Cow::Owned`].
    fn shape(&self) -> &Self::Shape;

    /// Index into this layout in "linear" fashion by moving across axes from left to right.
    fn index_linear_left(&self, idx: usize) -> isize;

    /// Index into this layout in "linear" fashion by moving across axes from right to left.
    fn index_linear_right(&self, idx: usize) -> isize;

    /// Index into this layout by traversing it in a memory order that is as efficient as possible.
    fn index_memory_order(&self, idx: usize) -> isize;

    /// Index into this layout with a multidimensional index.
    fn index(&self, idx: &Self::Index) -> isize;

    // fn first_index(&self) -> Option<Self::Index>;

    // fn next_for(&self, index: Self::Index) -> Option<Self::Index>;

    /// The number of elements between the minimum and maximum offsets represented by this layout.
    fn offset_range_checked(&self) -> Option<isize>;

    /// The number of bytes between the minimum and maximum offsets represented by this layout.
    ///
    /// If the number of bytes exceeds `isize::MAX`, returns `None`.
    ///
    /// # Safety
    /// This method must return `None` if and only if the number of bytes exceeds `isize::MAX`.
    /// When returning `Some(_)`, the number of bytes returned must be exact and correct.
    ///
    /// This method has a default implementation that uses [`Layout::offset_range_checked`],
    /// so that users should rarely need to re-implement this method.
    fn offset_range_bytes_checked<T>(&self) -> Option<isize>
    {
        size_of::<T>()
            .to_isize()
            .and_then(|s| self.offset_range_checked().map(|or| (s, or)))
            .and_then(|(s, or)| s.checked_mul(or))
    }

    // Shortcut methods

    fn size_checked(&self) -> Option<usize>
    {
        self.shape().size_checked()
    }

    fn size_bytes_checked<T>(&self) -> Option<usize>
    {
        self.shape().size_bytes_checked::<T>()
    }
}

pub trait Strided: Layout
{
    type Strides: Strides<Dimality = Self::Dimality>;

    fn strides(&self) -> &Self::Strides;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NLayout<const N: usize>
{
    shape: NShape<N>,
    strides: NStrides<N>,
}

impl<const N: usize> Default for NLayout<N>
where NDim<N>: Dimensionality
{
    fn default() -> Self
    {
        let shape = NShape::<N>::default_checked();
        let strides = NStrides::<N>::default_c(shape);
        Self {
            shape: shape.unwrap(),
            strides,
        }
    }
}

impl<const N: usize> Dimensioned for NLayout<N>
where NDim<N>: Dimensionality
{
    type Dimality = NDim<N>;

    fn ndim(&self) -> usize
    {
        N
    }
}

/// Get the offset between the minimum and maximum addresses of the strided array, in number of elements.
pub fn offset_range_checked(shape: &impl Shape, strides: &impl Strides) -> Option<isize>
{
    shape
        .iter_isize_checked()
        .and_then(|it| {
            it.zip(strides.iter())
                .map(|(sh, st)| (sh - 1).checked_mul(*st).map(|v| (v.min(0), v.max(0))))
                .try_fold((0isize, 0isize), |acc, x| x.map(|x_| (acc.0 + x_.0, acc.1 + x_.1)))
        })
        .and_then(|(min, max)| max.checked_sub(min))
}

pub fn offset_range_bytes_checked<T>(shape: &impl Shape, strides: &impl Strides) -> Option<isize>
{
    let size = size_of::<T>().to_isize();
    match size {
        None => None,
        Some(s) => offset_range_checked(shape, strides).and_then(|off| off.checked_mul(s)),
    }
}

impl<const N: usize> Layout for NLayout<N>
where NDim<N>: Dimensionality
{
    type Shape = NShape<N>;

    type Index = [usize; N];

    fn shape(&self) -> &Self::Shape
    {
        &self.shape
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

    fn index(&self, index: &Self::Index) -> isize
    {
        let mut offset = 0isize;
        for idx in 0..N {
            offset += (index[idx] as isize) * self.strides[idx];
        }
        offset
    }

    fn offset_range_checked(&self) -> Option<isize>
    {
        offset_range_checked(&self.shape, &self.strides)
    }

    fn offset_range_bytes_checked<T>(&self) -> Option<isize>
    {
        offset_range_bytes_checked::<T>(&self.shape, &self.strides)
    }

    // fn first_index(&self) -> Option<Self::Index>
    // {
    //     todo!()
    // }

    // fn next_for(&self, index: Self::Index) -> Option<Self::Index>
    // {
    //     todo!()
    // }
}

impl NLayout<1>
{
    pub fn new(length: usize) -> Self
    {
        NLayout {
            shape: [length].into(),
            strides: [1].into(),
        }
    }
}

macro_rules! create_L_types {
    ($(($t:ident, $v:literal)),+) => {
        $(
            pub type $t = NLayout<$v>;
        )+
    };
}

create_L_types!(
    (L1, 1),
    (L2, 2),
    (L3, 3),
    (L4, 4),
    (L5, 5),
    (L6, 6),
    (L7, 7),
    (L8, 8),
    (L9, 9),
    (L10, 10),
    (L11, 11),
    (L12, 12)
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynLayout
{
    shape: DShape,
    strides: DStrides,
}

impl Dimensioned for DynLayout
{
    type Dimality = DDyn;

    fn ndim(&self) -> usize
    {
        self.shape.ndim()
    }
}

impl Default for DynLayout
{
    fn default() -> Self
    {
        todo!()
        // let shape = DShape::Inline(1, []);
        // let strides = DStrides::default_c(shape);
        // Self {
        //     shape: Default::default(),
        //     strides: Default::default(),
        // }
    }
}

impl Layout for DynLayout
{
    type Shape = DShape;

    type Index = [usize];

    fn shape(&self) -> &Self::Shape
    {
        todo!()
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

    fn index(&self, idx: &Self::Index) -> isize
    {
        todo!()
    }

    fn offset_range_checked(&self) -> Option<isize>
    {
        offset_range_checked(&self.shape, &self.strides)
    }

    fn offset_range_bytes_checked<T>(&self) -> Option<isize>
    {
        let size = size_of::<T>().to_isize();
        match size {
            None => None,
            Some(s) => self
                .offset_range_checked()
                .and_then(|off| off.checked_mul(s)),
        }
    }

    // fn first_index(&self) -> Option<Self::Index>
    // {
    //     todo!()
    // }

    // fn next_for(&self, index: Self::Index) -> Option<Self::Index>
    // {
    //     todo!()
    // }
}

/// A convenience extension trait to check if a `usize` fits in an `isize`.
pub trait FitsInISize: Sized
{
    fn fits_in_isize(&self) -> bool;

    fn as_usize_if_isize_compatible(self) -> Option<Self>;
}

impl FitsInISize for usize
{
    fn fits_in_isize(&self) -> bool
    {
        *self <= (isize::MAX as usize)
    }

    fn as_usize_if_isize_compatible(self) -> Option<Self>
    {
        if self.fits_in_isize() {
            Some(self)
        } else {
            None
        }
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

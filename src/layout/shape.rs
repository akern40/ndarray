use alloc::borrow::Cow;
use core::{
    array::IntoIter,
    fmt::Debug,
    marker::PhantomData,
    ops::{Index, IndexMut},
};
use std::iter::Map;

use crate::{
    layout::{checked::Checked, FitsInISize},
    Axis,
};

use super::{DShape, Dimensioned, NShape};

use super::dimensionality::{DMax, Dimensionality, NDim};
use super::ShapeStrideError;

/// The trait for types that contain the shape of a multidimensional array.
///
/// To be specific, "shape" in this context for an `N`-dimensional array is a list of
/// `N` `usize` values, where the `i`th value describes the length of the `i`th dimension
/// of the `N`-dimensional array.
///
/// Types that implement `Shape` should not carry information about the strides
/// of an array, i.e., how the array's elements are laid out in memory. They only
/// inform the user about the lengths of each dimension.
///
/// # Constant Shapes
/// Having one or more axes of a shape be `const`-valued can be critical for array performance.
/// The `Shape` trait is agnostic as to whether any or all of the axes are `const` - as a result,
/// `Shape` is a read-only construct. See [`AxisMut`] for shapes that have a  non-`const` axis length.
/// See [`ShapeMut`] for shapes that don't have any `const` axis lengths.
///
/// # Safety
/// Although each dimension of a shape is a `usize`, not all shapes are valid.
/// In particular, arrays cannot have more than `isize::MAX` elements;
/// so any shape whose product is more than `isize::MAX` cannot represent an array
/// in-memory. The same goes for the number of bytes in an array.
/// Critically, _this trait does not guarantee that its implementing types
/// are always valid_. See [`CheckedShape`] for an unsafe trait that guarantees this property.
///
/// However, types implementing `Shape` must uphold the following safety guarantee:
/// [`Shape::size_checked`] must:
/// 1. Only return `Some(_)` if the total number of array elements is less than or equal to `isize::MAX`
/// 2. When returning `Some(_)`, the returned value must be exactly the number of expected elements
///
/// These two guarantees allow users of `Shape` to know that, when `size_checked` returns `Some(_)` they can:
/// 1. Cast all elements of the shape from `usize` to `isize`
/// 2. Ensure that an array constructed with the shape can be indexed by an `isize` offset
/// 3. Pre-allocate exactly that many elements
///
/// Similar guarantees must be upheld for [`Shape::size_bytes_checked`].
///
/// Both of these methods have reasonable default implementations using [`Shape::iter`],
/// so implementors of `Shape` won't frequently need to implement these functions.
pub unsafe trait Shape:
    Dimensioned + Index<usize, Output = usize> + Index<Axis, Output = usize> + Eq + Clone + Send + Sync + Debug
// + IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + ExactSizeIterator>
{
    /// The pattern matching-friendly form of the shape.
    ///
    /// This can be any type that allows for pattern matching.
    /// For the standard [`NShape`], this is:
    /// - `[usize]` for `NShape<1>`
    /// - `[usize, usize]` for `NShape<2>`
    /// - etc...
    ///
    /// and for the dynamic [`DShape`], this is just `DShape`
    type Pattern: IntoShape + Clone + Debug + PartialEq + Eq;

    type Iter<'a>: Iterator<Item = &'a usize> + ExactSizeIterator + DoubleEndedIterator
    where Self: 'a;

    /// Get the shape as a (possibly-borrowed) slice of `usize`.
    fn as_slice(&self) -> Cow<'_, [usize]>;

    /// Iterate over the dimensions of the shape.
    fn iter(&self) -> Self::Iter<'_>;

    /// Check the shape for overflow and wrap it in `Checked`
    fn into_checked(self) -> Option<Checked<Self>>
    {
        self.size_checked()
            .map(|_| unsafe { Checked::new_shape(self) })
    }

    fn into_checked_for<T>(self) -> Option<Checked<Self>>
    {
        match self.size_bytes_checked::<T>() {
            None => None,
            Some(_) => Some(unsafe { Checked::new_shape(self) }),
        }
    }

    /// Try to index the shape mutably, if `index` is a non-`const` axis.
    ///
    /// Types that implement `Shape` do not guarantee any sort of mutability; this method
    /// allows a non-panicking way to discover whether a given axis of a shape is mutable.
    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, ShapeStrideError<Self>>;

    /// Try to create an `ndim`-dimensional shape filled with `value`.
    ///
    /// This may fail if either the number of dimensions does not match the dimensionality
    /// of the shape, or if the shape has any `const` axis lengths.
    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>>;

    /// Get a pattern matching-friendly version of the shape.
    fn into_pattern(&self) -> Self::Pattern;

    /// Convert this shape into a dynamic-dimensional shape.
    fn to_dyn(&self) -> DShape
    {
        self.as_slice().into()
    }

    /// Get the number of elements that the array contains.
    ///
    /// If the number of elements is greater than `isize::MAX`, returns `None`.
    ///
    /// # Safety
    /// This method checks for overflow past `isize`, not `usize`. If this method returns `Some(_)`,
    /// then users know they can safely cast all dimensions of the shape to `isize`, which is often
    /// necessary for calculations with strides and offsets.
    fn size_checked(&self) -> Option<usize>
    {
        self.as_slice()
            .iter()
            .try_fold(1_usize, |acc, &i| acc.checked_mul(i))
            .and_then(FitsInISize::as_usize_if_isize_compatible)
    }

    /// Get the number of bytes that this array would fill.
    ///
    /// # Safety
    /// This method checks for bytes overflow past `isize`. If this method returns `Some(_)`,
    /// then users know that an allocated array is indexable using `isize` offsets.
    fn size_bytes_checked<T>(&self) -> Option<usize>
    {
        self.size_checked()
            .and_then(|v| v.checked_mul(size_of::<T>()))
            .and_then(FitsInISize::as_usize_if_isize_compatible)
    }

    fn iter_isize_checked<'a>(&'a self) -> Option<Map<Self::Iter<'a>, impl FnMut(&'a usize) -> isize>>
    {
        self.size_checked()
            .map(|_| self.iter().map(|v| *v as isize))
    }

    /// Try to turn this shape into a constant `N`-dimensional shape.
    fn try_to_nshape<const N: usize>(&self) -> Result<NShape<N>, ShapeStrideError<NShape<N>>>
    {
        if self.ndim() == N {
            let mut values = [0; N];
            values.split_at_mut(N).0.copy_from_slice(&self.as_slice());
            Ok(values.into())
        } else {
            Err(ShapeStrideError::BadDimality(PhantomData, self.ndim()))
        }
    }
}

/// A shape with fewer elements than `usize::MAX`.
///
/// # Safety
/// Although each dimension of a shape is a `usize`, not all shapes are valid.
/// In particular, arrays cannot have more than `isize::MAX` elements;
/// so any shape whose product is more than `isize::MAX` is invalid.
/// Implementing this trait guarantees that the implementing type
/// 1. When constructed, will have a product less than or equal to `isize::MAX`
/// 2. Cannot be modified to have a product greater than `isize::MAX`
pub unsafe trait CheckedShape: Shape
{
    /// Get the number of elements that the array contains.
    fn size(&self) -> usize
    {
        self.size_checked()
            .expect("Types implementing `CheckedShape` should have `Some(_)` size")
    }

    fn iter_isize<'a>(&'a self) -> impl Iterator<Item = isize> + DoubleEndedIterator + ExactSizeIterator + 'a
    {
        self.iter().map(|v| *v as isize)
    }
}

/// A shape whose `N`th dimension length is mutable.
pub trait AxisMut<const N: usize>: Shape
where NDim<N>: Dimensionality + DMax<Self::Dimality, Output = Self::Dimality>
{
    /// Get a mutable reference to the shape at the constant index.
    ///
    /// # Panics
    /// This function panics if it is called on a dynamic-dimensionality shape whose
    /// runtime dimensionality is less than `N`.
    fn get_mut(&mut self) -> &mut usize;
}

/// A shape with no `const`-valued axis lengths; i.e., a mutable shape.
///
/// Implementing `ShapeMut` will automatically implement [`AxisMut<N>`] for all `N` less than
/// the dimensionality of the shape. Note that dynamic-dimensionality shapes (i.e.,
/// `Shape<Dimality = DDyn>`) will implement `AxisMut` for all values of `N`.
pub trait ShapeMut: Shape + IndexMut<usize, Output = usize>
// + AddAssign
// + for<'a> AddAssign<&'a Self>
// + SubAssign
// + for<'a> SubAssign<&'a Self>
// + MulAssign
// + for<'a> MulAssign<&'a Self>
// + MulAssign<usize>
{
}

/// Implement `AxisMut` for axes less than the dimensionality.
///
/// This also implements `AxisMut` for all axes if the shape has a dynamic dimensionality.
impl<const N: usize, T> AxisMut<N> for T
where
    T: ShapeMut,
    NDim<N>: Dimensionality + DMax<T::Dimality, Output = T::Dimality>,
{
    fn get_mut(&mut self) -> &mut usize
    {
        &mut self[N]
    }
}

/// A conversion trait for types that can turn into a shape.
pub trait IntoShape
{
    type Dimality: Dimensionality;

    type Shape: Shape<Dimality = Self::Dimality>;

    fn into_shape(&self) -> Self::Shape;
}

impl<T> IntoShape for T
where T: Shape
{
    type Dimality = T::Dimality;

    type Shape = T;

    fn into_shape(&self) -> Self::Shape
    {
        self.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstMatrixShape<const N: usize, const M: usize>;

impl<const N: usize, const M: usize> Dimensioned for ConstMatrixShape<N, M>
{
    type Dimality = NDim<2>;

    fn ndim(&self) -> usize
    {
        2
    }
}

impl<const N: usize, const M: usize> Index<usize> for ConstMatrixShape<N, M>
{
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output
    {
        match index {
            0 => &N,
            1 => &M,
            _ => panic!("Index {index} out of bounds for ConstMatrixShape"),
        }
    }
}

impl<const N: usize, const M: usize> Index<Axis> for ConstMatrixShape<N, M>
{
    type Output = usize;

    fn index(&self, index: Axis) -> &Self::Output
    {
        self.index(index.0)
    }
}

impl<const N: usize, const M: usize> IntoIterator for ConstMatrixShape<N, M>
{
    type Item = usize;

    type IntoIter = IntoIter<usize, 2>;

    fn into_iter(self) -> Self::IntoIter
    {
        [N, M].into_iter()
    }
}

pub struct ConstMatrixIter<'a, const N: usize, const M: usize>
{
    start: usize,
    end: usize,
    _phantom: &'a PhantomData<usize>,
}

impl<'a, const N: usize, const M: usize> ConstMatrixIter<'a, N, M>
{
    fn new() -> Self
    {
        Self {
            start: 0,
            end: 2,
            _phantom: &PhantomData,
        }
    }
}

impl<'a, const N: usize, const M: usize> Iterator for ConstMatrixIter<'a, N, M>
{
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.start >= self.end {
            None
        } else if self.start == 0 {
            self.start += 1;
            Some(&N)
        } else {
            self.start += 1;
            Some(&M)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        (2, Some(2))
    }
}

impl<'a, const N: usize, const M: usize> DoubleEndedIterator for ConstMatrixIter<'a, N, M>
{
    fn next_back(&mut self) -> Option<Self::Item>
    {
        if self.start >= self.end {
            None
        } else if self.end == 2 {
            self.end -= 1;
            Some(&M)
        } else {
            self.end -= 1;
            Some(&N)
        }
    }
}

impl<'a, const N: usize, const M: usize> ExactSizeIterator for ConstMatrixIter<'a, N, M> {}

unsafe impl<const N: usize, const M: usize> Shape for ConstMatrixShape<N, M>
{
    type Pattern = [usize; 2];

    type Iter<'a> = ConstMatrixIter<'a, N, M>;

    fn into_pattern(&self) -> Self::Pattern
    {
        [N, M]
    }

    fn as_slice(&self) -> Cow<'_, [usize]>
    {
        Cow::Borrowed(&[N, M])
    }

    fn iter(&self) -> Self::Iter<'_>
    {
        ConstMatrixIter::new()
    }

    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, ShapeStrideError<Self>>
    {
        if index < 2 {
            Err(ShapeStrideError::FixedIndex(PhantomData, index))
        } else {
            Err(ShapeStrideError::OutOfBounds(PhantomData, index))
        }
    }

    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>>
    {
        if ndim != 2 {
            Err(ShapeStrideError::BadDimality(PhantomData, ndim))
        } else if value != N {
            Err(ShapeStrideError::FixedIndex(PhantomData, 0))
        } else if value != N {
            Err(ShapeStrideError::FixedIndex(PhantomData, 1))
        } else {
            Ok(Self)
        }
    }
}

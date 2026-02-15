use core::array;
use core::iter::Map;

use crate::layout::rank::{ConstRank, DynRank, Rank};
use crate::layout::ranked::Ranked;

pub trait Shape: Ranked
{
    type Iter: Iterator<Item = usize> + ExactSizeIterator + DoubleEndedIterator;

    /// A representation of the shape as a type that can be destructured.
    ///
    /// For example, for 2D shapes, this should be something like [usize; 2].
    /// For 3D shapes, [usize; 3], etc.
    /// For shapes whose rank isn't known at compile time, this can be the marker
    /// type [`DMarker`].
    type Pattern: Ranked<NDim = Self::NDim>;

    /// The length of the array along a given axis.
    fn axis_len(&self, axis: usize) -> usize;

    /// Iterate over the dimensions of the shape.
    fn iter(&self) -> Self::Iter;

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
        self.iter()
            .try_fold(1_usize, |acc, i| acc.checked_mul(i))
            .and_then(as_usize_if_isize_compatible)
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
            .and_then(as_usize_if_isize_compatible)
    }

    fn iter_isize<'a>(&'a self) -> Option<Map<Self::Iter, impl FnMut(usize) -> isize>>
    {
        self.size_checked().map(|_| self.iter().map(|v| v as isize))
    }
}

fn as_usize_if_isize_compatible(v: usize) -> Option<usize>
{
    if v <= (isize::MAX as usize) {
        Some(v)
    } else {
        None
    }
}

pub trait Patterned
{
    type Pattern;

    fn into_pattern(&self) -> Self::Pattern;
}

/// A conversion trait for types that can turn into a shape.
pub trait IntoShape
{
    type NDim: Rank;

    type Shape: Shape<NDim = Self::NDim>;

    fn into_shape(&self) -> Self::Shape;
}

impl<T> IntoShape for T
where T: Shape + Clone
{
    type NDim = T::NDim;

    type Shape = T;

    fn into_shape(&self) -> Self::Shape
    {
        self.clone()
    }
}

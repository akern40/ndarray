use core::iter::Map;

use crate::layout::ranked::Ranked;

/// A trait for array shapes.
pub trait Shape: Ranked
{
    /// The iterator type over the dimensions of the shape.
    type Iter<'a>: Iterator<Item = usize> + ExactSizeIterator + DoubleEndedIterator
    where Self: 'a;

    /// The length of the array along a given axis.
    fn axis_len(&self, axis: usize) -> usize;

    /// Iterate over the dimensions of the shape.
    fn iter(&self) -> Self::Iter<'_>;

    /// Get the number of elements that the array contains.
    ///
    /// If the number of elements is greater than `isize::MAX`, returns `None`.
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

    /// Iterate over the shape as `isize`.
    ///
    /// If the number of elements is greater than `isize::MAX`, returns `None`.
    fn iter_isize<'a>(&'a self) -> Option<Map<Self::Iter<'a>, impl FnMut(usize) -> isize>>
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

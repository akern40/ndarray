use alloc::borrow::Cow;
use core::{marker::PhantomData, ops::Index};

use crate::{
    layout::{shape::CheckedShape, Dimensioned, ShapeStrideError},
    Shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checked<T>(T);

impl<T> Checked<T>
{
    /// Create a new instance of `Checked` from an existing value.
    ///
    /// # Safety
    /// It is the caller's responsibility to ensure that the passed value
    /// already meets the invariants required by [`CheckedShape`].
    pub(crate) unsafe fn new_shape(value: T) -> Self
    where T: Shape
    {
        Self { 0: value }
    }

    pub(crate) fn unwrap(self) -> T
    {
        self.0
    }
}

impl<T> Dimensioned for Checked<T>
where T: Dimensioned
{
    type Dimality = T::Dimality;

    fn ndim(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<T> IntoIterator for Checked<T>
where T: IntoIterator
{
    type Item = T::Item;

    type IntoIter = T::IntoIter;

    fn into_iter(self) -> Self::IntoIter
    {
        self.0.into_iter()
    }
}

impl<T, I> Index<I> for Checked<T>
where T: Index<I>
{
    type Output = T::Output;

    fn index(&self, index: I) -> &Self::Output
    {
        self.0.index(index)
    }
}

unsafe impl<T> Shape for Checked<T>
where T: Shape
{
    type Pattern = T::Pattern;

    type Iter<'a>
        = T::Iter<'a>
    where Self: 'a;

    fn as_slice(&self) -> Cow<'_, [usize]>
    {
        self.0.as_slice()
    }

    fn iter(&self) -> Self::Iter<'_>
    {
        self.0.iter()
    }

    fn try_index_mut(&mut self, index: usize) -> Result<&mut usize, ShapeStrideError<Self>>
    {
        Err(ShapeStrideError::FixedIndex(PhantomData, index))
    }

    fn try_full(ndim: usize, value: usize) -> Result<Self, ShapeStrideError<Self>>
    {
        match ndim.checked_mul(value) {
            None => Err(ShapeStrideError::ShapeOverflow),
            Some(_) => T::try_full(ndim, value)
                .map(|v| unsafe { Checked::new_shape(v) })
                .map_err(|e| e.replace_type_with()),
        }
    }

    fn into_pattern(&self) -> Self::Pattern
    {
        self.0.into_pattern()
    }
}

unsafe impl<T> CheckedShape for Checked<T> where T: Shape {}

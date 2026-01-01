use crate::{layout::Strided, Axis, Dimension, Ixs, Layout};
use alloc::borrow::Cow;

/// Create a new Axes iterator
pub(crate) fn axes_of<'a, L>(layout: L) -> Axes<'a, L>
where L: Strided
{
    Axes {
        shape: layout.shape(),
        strides: layout.strides(),
        start: 0,
        end: layout.ndim(),
    }
}

/// An iterator over the length and stride of each axis of an array.
///
/// This iterator is created from the array method
/// [`.axes()`](crate::ArrayBase::axes).
///
/// Iterator element type is [`AxisDescription`].
///
/// # Examples
///
/// ```
/// use ndarray::Array3;
/// use ndarray::Axis;
///
/// let a = Array3::<f32>::zeros((3, 5, 4));
///
/// // find the largest axis in the array
/// // check the axis index and its length
///
/// let largest_axis = a.axes()
///                     .max_by_key(|ax| ax.len)
///                     .unwrap();
/// assert_eq!(largest_axis.axis, Axis(1));
/// assert_eq!(largest_axis.len, 5);
/// ```
#[derive(Debug, Clone)]
pub struct Axes<'a, L>
where L: Strided
{
    shape: Cow<'a, L::Shape>,
    strides: Cow<'a, L::Strides>,
    start: usize,
    end: usize,
}

/// Description of the axis, its length and its stride.
#[derive(Debug, Clone, Copy)]
pub struct AxisDescription
{
    /// Axis identifier (index)
    pub axis: Axis,
    /// Length in count of elements of the current axis
    pub len: usize,
    /// Stride in count of elements of the current axis
    pub stride: isize,
}

impl<D> Iterator for Axes<'_, D>
where D: Strided
{
    /// Description of the axis, its length and its stride.
    type Item = AxisDescription;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.start < self.end {
            let i = self.start.post_inc();
            Some(AxisDescription {
                axis: Axis(i),
                len: self.dim[i],
                stride: self.strides[i] as Ixs,
            })
        } else {
            None
        }
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where F: FnMut(B, AxisDescription) -> B
    {
        (self.start..self.end)
            .map(move |i| AxisDescription {
                axis: Axis(i),
                len: self.dim[i],
                stride: self.strides[i] as isize,
            })
            .fold(init, f)
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.end - self.start;
        (len, Some(len))
    }
}

impl<D> DoubleEndedIterator for Axes<'_, D>
where D: Dimension
{
    fn next_back(&mut self) -> Option<Self::Item>
    {
        if self.start < self.end {
            let i = self.end.pre_dec();
            Some(AxisDescription {
                axis: Axis(i),
                len: self.dim[i],
                stride: self.strides[i] as Ixs,
            })
        } else {
            None
        }
    }
}

trait IncOps: Copy
{
    fn post_inc(&mut self) -> Self;
    fn pre_dec(&mut self) -> Self;
}

impl IncOps for usize
{
    #[inline(always)]
    fn post_inc(&mut self) -> Self
    {
        let x = *self;
        *self += 1;
        x
    }
    #[inline(always)]
    fn pre_dec(&mut self) -> Self
    {
        *self -= 1;
        *self
    }
}

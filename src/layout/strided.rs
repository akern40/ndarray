//! The strided layout: an axis-length vector and an element-stride vector of
//! the same dimensionality.
//!
//! This is the representation `ArrayBase` has always used, moved into one
//! struct so that the array's layout can be handled as a single value. The
//! fields are deliberately kept as they were; this type does not (yet) have an
//! API of its own.

use crate::dimension::{self, Dimension, DimensionExt};
use crate::layout::ranked::Ranked;
use crate::layout::traits::{AxisOps, Layout, Strided};
use crate::{Axis, IxDyn, NdIndex, Slice};

/// A strided layout: axis lengths (`dim`) and element-count strides
/// (`strides`, to be parsed as `isize`) of the same dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedLayout<D>
{
    /// The lengths of the axes.
    pub(crate) dim: D,
    /// The element count stride per axis. To be parsed as `isize`.
    pub(crate) strides: D,
}

impl<D> StridedLayout<D>
{
    pub(crate) const fn new(dim: D, strides: D) -> Self
    {
        Self { dim, strides }
    }
}

impl<D: Dimension> StridedLayout<D>
{
    /// The offset of `index`, or `None` if it is out of bounds.
    #[inline]
    pub(crate) fn index_offset_checked<I: NdIndex<D>>(&self, index: &I) -> Option<isize>
    {
        index.index_checked(&self.dim, &self.strides)
    }

    /// The offset of `index`, without bounds checking.
    #[inline]
    pub(crate) fn index_offset<I: NdIndex<D>>(&self, index: &I) -> isize
    {
        index.index_unchecked(&self.strides)
    }
}

impl<D: Dimension> Ranked for StridedLayout<D>
{
    type NDim = D::NDim;

    #[inline]
    fn ndim(&self) -> usize
    {
        self.dim.ndim()
    }
}

impl<D: Dimension> Layout for StridedLayout<D>
{
    type Index = D;

    #[inline]
    fn shape(&self) -> &[usize]
    {
        self.dim.slice()
    }

    #[inline]
    fn len(&self) -> usize
    {
        self.dim.size()
    }

    #[inline]
    #[track_caller]
    fn len_of(&self, axis: Axis) -> usize
    {
        self.dim[axis.index()]
    }

    #[inline]
    fn offset(&self, index: &D) -> isize
    {
        D::stride_offset(index, &self.strides)
    }

    #[inline]
    fn offset_checked(&self, index: &D) -> Option<isize>
    {
        self.dim.stride_offset_checked(&self.strides, index)
    }

    #[inline]
    fn first_index(&self) -> Option<D>
    {
        self.dim.first_index()
    }

    #[inline]
    fn next_index(&self, index: D) -> Option<D>
    {
        self.dim.next_for(index)
    }
}

impl<D: Dimension> Strided for StridedLayout<D>
{
    #[inline]
    fn strides(&self) -> &[isize]
    {
        let s = self.strides.slice();
        // strides are stored as usize and reinterpreted as isize
        unsafe { core::slice::from_raw_parts(s.as_ptr() as *const isize, s.len()) }
    }

    #[inline]
    #[track_caller]
    fn stride_of(&self, axis: Axis) -> isize
    {
        self.strides[axis.index()] as isize
    }

    #[inline]
    fn is_contiguous(&self) -> bool
    {
        D::is_contiguous(&self.dim, &self.strides)
    }

    #[inline]
    fn is_standard_layout(&self) -> bool
    {
        dimension::is_layout_c(&self.dim, &self.strides)
    }

    #[inline]
    fn is_fortran_layout(&self) -> bool
    {
        dimension::is_layout_f(&self.dim, &self.strides)
    }

    #[inline]
    fn low_addr_offset(&self) -> usize
    {
        dimension::offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides)
    }

    #[inline]
    fn has_overlap(&self) -> bool
    {
        dimension::dim_stride_overlap(&self.dim, &self.strides)
    }
}

impl<D: Dimension> AxisOps for StridedLayout<D>
{
    type Smaller = StridedLayout<D::Smaller>;
    type Larger = StridedLayout<D::Larger>;
    type Dyn = StridedLayout<IxDyn>;

    #[inline]
    #[track_caller]
    fn collapse_axis(&mut self, axis: Axis, index: usize) -> isize
    {
        dimension::do_collapse_axis(&mut self.dim, &self.strides, axis.index(), index)
    }

    #[inline]
    #[track_caller]
    fn slice_axis(&mut self, axis: Axis, slice: Slice) -> isize
    {
        dimension::do_slice(&mut self.dim.slice_mut()[axis.index()], &mut self.strides.slice_mut()[axis.index()], slice)
    }

    #[inline]
    #[track_caller]
    fn invert_axis(&mut self, axis: Axis) -> isize
    {
        let s = self.strides.axis(axis) as isize;
        let m = self.dim.axis(axis);
        self.strides.set_axis(axis, (-s) as usize);
        if m != 0 {
            dimension::stride_offset(m - 1, s as usize)
        } else {
            0
        }
    }

    #[inline]
    #[track_caller]
    fn swap_axes(&mut self, a: usize, b: usize)
    {
        self.dim.slice_mut().swap(a, b);
        self.strides.slice_mut().swap(a, b);
    }

    #[inline]
    fn reverse_axes(&mut self)
    {
        self.dim.slice_mut().reverse();
        self.strides.slice_mut().reverse();
    }

    #[inline]
    #[track_caller]
    fn merge_axes(&mut self, take: Axis, into: Axis) -> bool
    {
        dimension::merge_axes(&mut self.dim, &mut self.strides, take, into)
    }

    #[inline]
    #[track_caller]
    fn insert_axis(&self, axis: Axis) -> Self::Larger
    {
        assert!(axis.index() <= self.dim.ndim());
        StridedLayout::new(self.dim.insert_axis(axis), self.strides.insert_axis(axis))
    }

    #[inline]
    #[track_caller]
    fn remove_axis(&self, axis: Axis) -> Self::Smaller
    {
        assert!(self.dim.ndim() > 0, "cannot remove an axis from a 0-dimensional layout");
        assert!(axis.index() < self.dim.ndim());
        StridedLayout::new(self.dim.try_remove_axis(axis), self.strides.try_remove_axis(axis))
    }

    #[inline]
    #[track_caller]
    fn permute_axes(&self, axes: &D) -> Self
    {
        let n = self.dim.ndim();
        assert_eq!(axes.ndim(), n, "the permutation must list every axis");
        let mut usage_counts = D::zeros(n);
        for axis in axes.slice() {
            usage_counts[*axis] += 1;
        }
        for count in usage_counts.slice() {
            assert_eq!(*count, 1, "each axis must be listed exactly once");
        }
        let mut new_dim = usage_counts; // reuse to avoid an allocation
        let mut new_strides = D::zeros(n);
        for (new_axis, &axis) in axes.slice().iter().enumerate() {
            new_dim[new_axis] = self.dim[axis];
            new_strides[new_axis] = self.strides[axis];
        }
        StridedLayout::new(new_dim, new_strides)
    }

    #[inline]
    fn into_dyn(self) -> Self::Dyn
    {
        StridedLayout::new(self.dim.into_dyn(), self.strides.into_dyn())
    }
}

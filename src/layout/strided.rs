//! The strided layout: an axis-length vector and an element-stride vector of
//! the same dimensionality.
//!
//! This is the representation `ArrayBase` has always used, moved into one
//! struct so that the array's layout can be handled as a single value. The
//! fields are deliberately kept as they were; this type does not (yet) have an
//! API of its own.

use crate::dimension::{self, Dimension};
use crate::layout::ranked::Ranked;
use crate::layout::traits::{Layout, Strided};
use crate::Axis;

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

impl<D: Dimension> Ranked for StridedLayout<D>
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.dim.ndim()
    }
}

impl<D: Dimension> Layout for StridedLayout<D>
{
    type Index = D;

    fn shape(&self) -> &[usize]
    {
        self.dim.slice()
    }

    fn len(&self) -> usize
    {
        self.dim.size()
    }

    #[track_caller]
    fn len_of(&self, axis: Axis) -> usize
    {
        self.dim[axis.index()]
    }

    fn offset(&self, index: &D) -> isize
    {
        D::stride_offset(index, &self.strides)
    }

    fn offset_checked(&self, index: &D) -> Option<isize>
    {
        self.dim.stride_offset_checked(&self.strides, index)
    }

    fn first_index(&self) -> Option<D>
    {
        self.dim.first_index()
    }

    fn next_index(&self, index: D) -> Option<D>
    {
        self.dim.next_for(index)
    }
}

impl<D: Dimension> Strided for StridedLayout<D>
{
    fn strides(&self) -> &[isize]
    {
        let s = self.strides.slice();
        // strides are stored as usize and reinterpreted as isize
        unsafe { core::slice::from_raw_parts(s.as_ptr() as *const isize, s.len()) }
    }

    #[track_caller]
    fn stride_of(&self, axis: Axis) -> isize
    {
        self.strides[axis.index()] as isize
    }

    fn is_contiguous(&self) -> bool
    {
        D::is_contiguous(&self.dim, &self.strides)
    }

    fn is_standard_layout(&self) -> bool
    {
        dimension::is_layout_c(&self.dim, &self.strides)
    }

    fn is_fortran_layout(&self) -> bool
    {
        dimension::is_layout_f(&self.dim, &self.strides)
    }

    fn low_addr_offset(&self) -> usize
    {
        dimension::offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides)
    }

    fn has_overlap(&self) -> bool
    {
        dimension::dim_stride_overlap(&self.dim, &self.strides)
    }
}

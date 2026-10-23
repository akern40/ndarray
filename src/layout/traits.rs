//! The layout traits: what an array needs from "the map from logical index to
//! element offset", split by how much the layout promises.
//!
//! * [`Layout`]: shape, size and the index → offset map. Enough for indexing
//!   and element iteration.
//! * [`Strided`]: the layout is a stride vector, so strides, contiguity and
//!   memory-order questions have answers.
//!
//! Methods are added as `ArrayBase`'s own implementation needs them; nothing
//! here is a complete API yet.

use crate::dimension::Dimension;
use crate::layout::ranked::Ranked;
use crate::Axis;

/// The map from logical index to element offset, plus the shape.
pub trait Layout: Ranked + Clone
{
    /// The index type this layout is addressed with.
    type Index: Dimension;

    /// The axis lengths.
    fn shape(&self) -> &[usize];

    /// The number of elements: the product of the axis lengths.
    fn len(&self) -> usize;

    /// Whether the layout has no elements (some axis has length 0).
    fn is_empty(&self) -> bool
    {
        self.len() == 0
    }

    /// The length of `axis`.
    ///
    /// ***Panics*** if the axis is out of bounds.
    fn len_of(&self, axis: Axis) -> usize;

    /// The element offset of `index`, without bounds checking.
    fn offset(&self, index: &Self::Index) -> isize;

    /// The element offset of `index`, or `None` if it is out of bounds.
    fn offset_checked(&self, index: &Self::Index) -> Option<isize>;

    /// The first index in logical (C) order, or `None` if the layout is empty.
    fn first_index(&self) -> Option<Self::Index>;

    /// The index following `index` in logical (C) order, or `None` at the end.
    fn next_index(&self, index: Self::Index) -> Option<Self::Index>;
}

/// A layout that is a stride vector: `offset(index) = Σ index[k] * stride[k]`.
pub trait Strided: Layout
{
    /// The element strides, one per axis.
    fn strides(&self) -> &[isize];

    /// The stride of `axis`.
    ///
    /// ***Panics*** if the axis is out of bounds.
    fn stride_of(&self, axis: Axis) -> isize;

    /// Whether the elements occupy one contiguous block in memory, in any
    /// axis order.
    fn is_contiguous(&self) -> bool;

    /// Whether the layout is contiguous in logical (C / row-major) order.
    fn is_standard_layout(&self) -> bool;

    /// Whether the layout is contiguous in Fortran (column-major) order.
    fn is_fortran_layout(&self) -> bool;

    /// Offset from the lowest-addressed element to the element at index 0.
    fn low_addr_offset(&self) -> usize;

    /// Whether two distinct in-bounds indices can map to the same offset.
    fn has_overlap(&self) -> bool;
}

//! The layout traits: what an array needs from "the map from logical index to
//! element offset", split by how much the layout promises.
//!
//! * [`Layout`]: shape, size and the index → offset map. Enough for indexing
//!   and element iteration.
//! * [`AxisOps`]: the layout is made of axes that can be indexed, sliced,
//!   inserted, removed, permuted and merged. In-place operations that move
//!   the origin return the element offset the array's handle must advance by.
//! * [`Strided`]: the layout is a stride vector, so strides, contiguity and
//!   memory-order questions have answers.
//!
//! Methods are added as `ArrayBase`'s own implementation needs them; nothing
//! here is a complete API yet.

use crate::dimension::Dimension;
use crate::layout::ranked::Ranked;
use crate::{Axis, Slice};

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

/// A layout whose axes are first-class: they can be indexed, sliced,
/// inserted, removed, permuted and merged without touching the elements.
///
/// Operations that change the origin (`collapse_axis`, `slice_axis`,
/// `invert_axis`) return the offset, in elements, by which the array's handle
/// must advance so that logical index 0 still names the right element.
pub trait AxisOps: Layout
{
    /// This layout with one axis removed.
    type Smaller: AxisOps;
    /// This layout with one axis added.
    type Larger: AxisOps;
    /// This layout with dynamic rank.
    type Dyn: AxisOps;

    /// Fix `axis` at `index`, leaving it with length 1; returns the origin offset.
    ///
    /// ***Panics*** if `index` is out of bounds.
    fn collapse_axis(&mut self, axis: Axis, index: usize) -> isize;

    /// Restrict `axis` to `slice`; returns the origin offset.
    ///
    /// ***Panics*** if the slice is out of bounds.
    fn slice_axis(&mut self, axis: Axis, slice: Slice) -> isize;

    /// Reverse the direction of `axis`; returns the origin offset.
    fn invert_axis(&mut self, axis: Axis) -> isize;

    /// Swap two axes.
    fn swap_axes(&mut self, a: usize, b: usize);

    /// Reverse the order of all axes.
    fn reverse_axes(&mut self);

    /// Merge `take` into `into` if the result is still expressible; returns
    /// whether it was.
    fn merge_axes(&mut self, take: Axis, into: Axis) -> bool;

    /// Insert a length-1 axis at `axis`.
    ///
    /// ***Panics*** if `axis.index() > ndim`.
    fn insert_axis(&self, axis: Axis) -> Self::Larger;

    /// Remove `axis` (its length must already be 1 if the elements are to be
    /// preserved; see `collapse_axis`).
    ///
    /// ***Panics*** if the axis is out of bounds or the rank is 0.
    fn remove_axis(&self, axis: Axis) -> Self::Smaller;

    /// Reorder the axes: new axis `k` is old axis `axes[k]`.
    ///
    /// ***Panics*** unless `axes` is a permutation of `0..ndim`.
    fn permute_axes(&self, axes: &Self::Index) -> Self;

    /// Forget the static rank.
    fn into_dyn(self) -> Self::Dyn;
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

//! The strided layout: an axis-length vector and an element-stride vector of
//! the same dimensionality.
//!
//! This is the representation `ArrayBase` has always used, moved into one
//! struct so that the array's layout can be handled as a single value. The
//! fields are deliberately kept as they were; this type does not (yet) have an
//! API of its own.

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

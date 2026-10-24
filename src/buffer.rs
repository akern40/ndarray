//! Buffers: the thing a storage owns or borrows, one type per format.
//!
//! A buffer type identifies a *format* (dense, sparse, ...). Ownership of a
//! buffer — owned, shared, borrowed, copy-on-write — is a separate axis,
//! provided once for every format by the wrappers in [`crate::storage`]
//! (`Owned<B>`, `Shared<B>`, `View<'a, B>`, ...). Dependencies run one way:
//! storage → buffer → handle.

use crate::StridedLayout;
use crate::{ArrayBase, Dimension, Handle, Owned, View};

/// A buffer of elements in some format, addressed through a [`Handle`].
///
/// # Safety
///
/// Handles obtained from a buffer must stay valid when the buffer *value* is
/// moved (the elements must live behind a stable heap allocation), and
/// `origin`/`offset_of` must be inverses: `origin().offset(offset_of(h)) == h`
/// for every in-bounds `h`, also across a clone of the buffer.
pub unsafe trait Buffer: Sized
{
    /// The element type.
    type Elem;
    /// The handle type for this format.
    type Handle: Handle<Elem = Self::Elem>;

    /// Handle to the element at offset 0.
    fn origin(&self) -> Self::Handle;

    /// Element offset of `handle` from `origin()`.
    fn offset_of(&self, handle: Self::Handle) -> isize;

    /// Whether `handle` addresses this buffer (or one past its end).
    fn is_inbounds(&self, handle: Self::Handle) -> bool;

    /// Number of element slots the buffer spans.
    fn len(&self) -> usize;

    /// Whether the buffer spans no elements.
    fn is_empty(&self) -> bool
    {
        self.len() == 0
    }

    /// A deep copy of the buffer. Clonable exactly when the elements are.
    fn clone_buf(&self) -> Self
    where Self::Elem: Clone;

    /// Overwrite `self` with a deep copy of `other`, reusing `self`'s
    /// allocation where possible.
    fn clone_buf_from(&mut self, other: &Self)
    where Self::Elem: Clone
    {
        *self = other.clone_buf();
    }

    /// Copy the elements visible through `view` into a fresh buffer of the
    /// same format, in standard layout.
    fn collect<D: Dimension>(
        view: ArrayBase<View<'_, Self>, StridedLayout<D>>,
    ) -> ArrayBase<Owned<Self>, StridedLayout<D>>
    where Self::Elem: Clone;
}

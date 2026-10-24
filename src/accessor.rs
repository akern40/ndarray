//! Accessors: how an element offset produced by a layout is turned into an
//! element.
//!
//! An accessor is a *family* of storage: everything that stores its elements
//! the same way (dense buffer, sparse entries, ...) shares one accessor, and
//! the ownership kinds (owned, view, shared) of that family are its storages.
//! `ArrayBase` keeps a [`Accessor::Handle`] next to the layout; the storage
//! owns or borrows the buffer the handle points into.

use crate::{ArrayBase, Dimension, OwnedRepr, RawData, ViewRepr};
use core::fmt;
use core::marker::PhantomData;
use core::ptr::NonNull;

/// Maps element offsets to elements for one storage family.
///
/// # Safety
///
/// `offset` must be the only way an array moves its handle, and the handle
/// returned for an in-bounds offset must address the element at that offset.
pub unsafe trait Accessor: Sized
{
    /// The element type.
    type Elem;
    /// The value an array holds to reach its elements: a data pointer for
    /// dense storage.
    type Handle: Copy;

    /// Advance `handle` by `by` elements.
    ///
    /// # Safety
    ///
    /// The resulting handle must stay within (or one past) the buffer.
    unsafe fn offset(handle: Self::Handle, by: isize) -> Self::Handle;

    /// The element at `offset` from `handle`.
    ///
    /// # Safety
    ///
    /// `offset` must be in bounds of the buffer and the element initialized;
    /// the returned borrow must not alias a live `&mut` to the same element.
    unsafe fn get<'a>(handle: Self::Handle, offset: isize) -> &'a Self::Elem;

    /// The element at `offset` from `handle`, mutably.
    ///
    /// # Safety
    ///
    /// As [`get`](Self::get), and the caller must have unique access to the
    /// element (the layout must be unique and the storage uniquely owned).
    unsafe fn get_mut<'a>(handle: Self::Handle, offset: isize) -> &'a mut Self::Elem;

    /// The owning storage of this accessor family.
    type Owned: RawData<Elem = Self::Elem, Acc = Self>;

    /// Materialize a view of this family into its owning storage.
    fn into_owned<D>(view: ArrayBase<ViewRepr<&'_ Self>, D>) -> ArrayBase<Self::Owned, D>
    where
        Self::Elem: Clone,
        D: Dimension;
}

/// The dense family: elements in one contiguous allocation, addressed by a
/// data pointer. This is every storage ndarray has today.
pub struct Dense<A>(PhantomData<A>);

impl<A> fmt::Debug for Dense<A>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        f.write_str("Dense")
    }
}

unsafe impl<A> Accessor for Dense<A>
{
    type Elem = A;
    type Handle = NonNull<A>;

    #[inline(always)]
    unsafe fn offset(handle: NonNull<A>, by: isize) -> NonNull<A>
    {
        NonNull::new_unchecked(handle.as_ptr().offset(by))
    }

    #[inline(always)]
    unsafe fn get<'a>(handle: NonNull<A>, offset: isize) -> &'a A
    {
        &*handle.as_ptr().offset(offset)
    }

    #[inline(always)]
    unsafe fn get_mut<'a>(handle: NonNull<A>, offset: isize) -> &'a mut A
    {
        &mut *handle.as_ptr().offset(offset)
    }

    type Owned = OwnedRepr<A>;

    fn into_owned<D>(view: ArrayBase<ViewRepr<&'_ Self>, D>) -> ArrayBase<OwnedRepr<A>, D>
    where
        A: Clone,
        D: Dimension,
    {
        view.to_owned()
    }
}

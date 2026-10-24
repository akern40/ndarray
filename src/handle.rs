//! Handles: the value an array holds to reach its elements, and how an
//! element offset produced by a layout is turned into an element.
//!
//! A handle type identifies a *family* of storage: everything that stores its
//! elements the same way (dense buffer, sparse entries, ...) shares one handle
//! type, and the ownership kinds (owned, view, shared) of that family are its
//! storages. `ArrayBase` keeps a handle next to the layout; the storage owns or
//! borrows the buffer the handle points into.

use core::fmt;
use core::ptr::NonNull;

/// Maps element offsets to elements for one storage family.
///
/// # Safety
///
/// `offset` must be the only way an array moves its handle, and the handle
/// returned for an in-bounds offset must address the element at that offset.
pub unsafe trait Handle: Copy + fmt::Debug
{
    /// The element type.
    type Elem;

    /// Advance the handle by `by` elements.
    ///
    /// # Safety
    ///
    /// The resulting handle must stay within (or one past) the buffer.
    unsafe fn offset(self, by: isize) -> Self;

    /// The element at `offset` from this handle.
    ///
    /// # Safety
    ///
    /// `offset` must be in bounds of the buffer and the element initialized;
    /// the returned borrow must not alias a live `&mut` to the same element.
    unsafe fn get<'a>(self, offset: isize) -> &'a Self::Elem;

    /// The element at `offset` from this handle, mutably.
    ///
    /// # Safety
    ///
    /// As [`get`](Self::get), and the caller must have unique access to the
    /// element (the layout must be unique and the storage uniquely owned).
    unsafe fn get_mut<'a>(self, offset: isize) -> &'a mut Self::Elem;
}

/// The dense family: elements in one contiguous allocation, addressed by a
/// data pointer. This is every storage ndarray has today.
unsafe impl<A> Handle for NonNull<A>
{
    type Elem = A;

    #[inline(always)]
    unsafe fn offset(self, by: isize) -> NonNull<A>
    {
        NonNull::new_unchecked(self.as_ptr().offset(by))
    }

    #[inline(always)]
    unsafe fn get<'a>(self, offset: isize) -> &'a A
    {
        &*self.as_ptr().offset(offset)
    }

    #[inline(always)]
    unsafe fn get_mut<'a>(self, offset: isize) -> &'a mut A
    {
        &mut *self.as_ptr().offset(offset)
    }
}

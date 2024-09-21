//! The core types of `ndarray`, redefined in terms of the new infrastructure.
//!
//! The idea is partially that the `core` module / folder could be extracted
//! into its own crate, but these types and implementations would remain here.

mod backend;
mod array;
mod arrayref;

use core::{mem, ptr::NonNull};
use std::sync::Arc;

pub use array::*;
pub use arrayref::*;
pub use backend::*;

use crate::Dimension;

/// Type alias for a dynamically-allocated, unique backend.
pub type VecBackend<T> = (VecOwner<T>, NonNull<T>);

/// This type should act almost entirely equivalently to the existing `Array`
pub type Array<A, D> = OwnedBase<D, VecBackend<A>>;

/// Completely equivalent to [`crate::OwnedRepr`]!
pub struct VecOwner<T>
{
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

/// The glue implementation for this backend.
unsafe impl<E> Backend for VecBackend<E>
{
    type Elem = E;

    type Ref = NonNull<E>;

    type Owned = VecOwner<E>;

    fn ref_from_owner_offset(owner: &Self::Owned, offset: isize) -> Self::Ref
    {
        todo!("unsafe {{ owner.ptr.offset(offset) }}")
    }
}

/// A very simple Uniqueable, since this type is always unique.
unsafe impl<A, D> Uniqueable for Array<A, D>
{
    fn try_ensure_unique(&mut self) {}

    fn try_is_unique(&self) -> Option<bool>
    {
        Some(true)
    }
}

/// A blanket NdArray implementation for any backend whose reference type is `NonNull<A>`.
///
/// As described in the `arrayref` module of `core`, this kind of reference-focused `impl`
/// is very ergonomic and makes sense, since most behavior relies on the reference type,
/// not the owned type.
unsafe impl<L, B, A> NdArray<B> for OwnedBase<L, B>
where B: Backend<Ref = NonNull<A>, Elem = A>
{
    fn as_ptr(&self) -> *mut <B as Backend>::Elem
    {
        self.aref.storage.as_ptr()
    }
}

/// And now our Arc-based backend
pub type ArcBackend<T> = (Arc<VecOwner<T>>, NonNull<T>);

/// Again, should be almost entirely equivalent to the existing `ArcArray`
pub type ArcArray<A, D> = OwnedBase<D, ArcBackend<A>>;

/// A simple backend implementation
unsafe impl<E> Backend for ArcBackend<E>
{
    type Elem = E;

    type Ref = NonNull<E>;

    type Owned = Arc<VecOwner<E>>;

    fn ref_from_owner_offset(owner: &Self::Owned, offset: isize) -> Self::Ref
    {
        todo!("unsafe {{ owner.as_ref().ptr.offset(isize) }}")
    }
}

// Uniqueable implementation, with the uniqueness logic stolen from what already exists
unsafe impl<A, D: Dimension> Uniqueable for ArcArray<A, D>
{
    fn try_ensure_unique(&mut self)
    {
        if Arc::get_mut(&mut self.own).is_some() {
            return;
        }
        if self.layout.size() <= self.own.len / 2 {
            todo!(".to_owned().to_shared()");
        }
        let ptr = self.as_ptr();
        let rcvec = &mut self.own;
        let a_size = mem::size_of::<A>() as isize;
        let our_off = if a_size != 0 {
            (ptr as isize - Arc::as_ptr(&rcvec) as isize) / a_size
        } else {
            0
        };
        self.storage = ArcBackend::<A>::ref_from_owner_offset(&mut self.own, our_off);
    }

    // We're using strong count here so that we don't need &mut self.
    // This is not as strong as the original `try_is_unique`, but it doesn't matter.
    // The original does not hold on to the mutable reference provided by [`Arc::get_mut`],
    // so the moment the call is over the uniqueness guarantee does not hold.
    // This is a weaker guarantee of instantaneous uniqueness, but neither this nor the
    // original implementation are good enough to safely do anything with that uniqueness anyway.
    fn try_is_unique(&self) -> Option<bool>
    {
        Some(Arc::strong_count(&self.own) == 1)
    }
}

/// These aliases work for a view into any array whose reference type is `NonNull`:

pub type ArrayView<'a, A, D> = ViewBase<'a, D, NonNull<A>, A>;
pub type ArrayViewMut<'a, A, D> = ViewBaseMut<'a, D, NonNull<A>, A>;
pub type RawArrayView<A, D> = RawViewBase<D, NonNull<A>, A>;
pub type RawArrayViewMut<A, D> = RawViewBaseMut<D, NonNull<A>, A>;
pub type RawArrayRef<A, D> = RawRefBase<D, NonNull<A>>;
pub type ArrayRef<A, D> = RefBase<D, NonNull<A>>;

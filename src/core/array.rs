//! Owning array types.

use core::{fmt::Debug, marker::PhantomData};

use super::{
    arrayref::{RawRefBase, RefBase},
    backend::Backend,
};

/// Base type for arrays with owning semantics.
pub struct OwnedBase<L, B: Backend>
{
    pub(super) aref: RefBase<L, B::Ref>,
    pub(super) own: B::Owned,
}

/// Base type for array views.
///
/// Views are like references; in fact, they're just wrappers for references.
/// The difference is that a reference's layout must be identical to the array
/// from which is has been derived; a view's layout may be different, representing
/// a segment, strided, or otherwise incomplete look at its parent array.
#[derive(Debug)]
pub struct ViewBase<'a, L, R, E>
{
    aref: RefBase<L, R>,
    life: PhantomData<&'a E>,
}

/// Base type for array views with mutable data.
///
/// All kinds of views can have their layout mutated. However, data mutation
/// is tracked separately via two different types.
#[derive(Debug)]
pub struct ViewBaseMut<'a, L, R, E>
{
    aref: RefBase<L, R>,
    life: PhantomData<&'a mut E>,
}

/// Base type for array views without lifetimes.
#[derive(Debug)]
pub struct RawViewBase<L, R, E>
{
    rref: RawRefBase<L, R>,
    life: PhantomData<*const E>,
}

/// Base type for array views without lifetimes, but with mutable data.
#[derive(Debug)]
pub struct RawViewBaseMut<L, R, E>
{
    rref: RawRefBase<L, R>,
    life: PhantomData<*const E>,
}

/// A trait for arrays with mutable data that can be made unique.
///
/// Essentially all monomorphizations of [`OwnedBase`], [`ViewBaseMut`],
/// and [`RawViewBaseMut`] should implement Uniqueable; this applies even
/// when the array type does not have any data sharing capabilities.
///
/// There are already blanket implementations for `ViewBaseMut` and
/// `RawViewBaseMut`; as a result, any creation of these types
/// _must_ ensure that the underlying data is unique (a.k.a, unshared)
/// before creating these mutable views.
pub unsafe trait Uniqueable
{
    fn try_ensure_unique(&mut self);

    fn try_is_unique(&self) -> Option<bool>;
}

/// Trait implemented by all the non-reference array types.
///
/// We'll probably want to split trait into multiple trait, for three reasons:
/// 1. Adding a "Raw" and "Mut" variants will likely be necessary
/// 2. We probably don't want a single trait specified by backend, but instead traits
/// that are specified by reference type, owned type, and backend separately.
/// This allows for blanket implementations that would otherwise be annoying.
/// 3. We may want to add subtraits that break the behavior into logical pieces,
/// instead of having a monolith.
pub unsafe trait NdArray<B: Backend>
{
    fn as_ptr(&self) -> *mut B::Elem;
}

mod array_impls
{
    use core::fmt::Debug;
    use core::ops::{Deref, DerefMut};

    use crate::core::{Backend, RawRefBase, RefBase};

    use super::{OwnedBase, RawViewBase, RawViewBaseMut, Uniqueable, ViewBase, ViewBaseMut};

    impl<L, B: Backend> Deref for OwnedBase<L, B>
    {
        type Target = RefBase<L, B::Ref>;

        fn deref(&self) -> &Self::Target
        {
            &self.aref
        }
    }

    impl<L, B: Backend> DerefMut for OwnedBase<L, B>
    where Self: Uniqueable
    {
        fn deref_mut(&mut self) -> &mut Self::Target
        {
            self.try_ensure_unique();
            &mut self.aref
        }
    }

    impl<'a, L, R, E> Deref for ViewBase<'a, L, R, E>
    {
        type Target = RefBase<L, R>;

        fn deref(&self) -> &Self::Target
        {
            &self.aref
        }
    }

    impl<'a, L, R, E> Deref for ViewBaseMut<'a, L, R, E>
    {
        type Target = RefBase<L, R>;

        fn deref(&self) -> &Self::Target
        {
            &self.aref
        }
    }

    impl<'a, L, R, E> DerefMut for ViewBaseMut<'a, L, R, E>
    {
        fn deref_mut(&mut self) -> &mut Self::Target
        {
            &mut self.aref
        }
    }

    impl<L, R, E> Deref for RawViewBase<L, R, E>
    {
        type Target = RawRefBase<L, R>;

        fn deref(&self) -> &Self::Target
        {
            &self.rref
        }
    }

    impl<L, R, E> Deref for RawViewBaseMut<L, R, E>
    {
        type Target = RawRefBase<L, R>;

        fn deref(&self) -> &Self::Target
        {
            &self.rref
        }
    }

    impl<L, R, E> DerefMut for RawViewBaseMut<L, R, E>
    {
        fn deref_mut(&mut self) -> &mut Self::Target
        {
            &mut self.rref
        }
    }

    unsafe impl<'a, L, R, E> Uniqueable for ViewBaseMut<'a, L, R, E>
    {
        fn try_ensure_unique(&mut self) {}

        fn try_is_unique(&self) -> Option<bool>
        {
            Some(true)
        }
    }

    unsafe impl<L, R, E> Uniqueable for RawViewBaseMut<L, R, E>
    {
        fn try_ensure_unique(&mut self) {}

        fn try_is_unique(&self) -> Option<bool>
        {
            None
        }
    }

    impl<L, B> Debug for OwnedBase<L, B>
    where
        B: Backend + Debug,
        B::Ref: Debug,
        B::Owned: Debug,
        L: Debug,
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
        {
            f.debug_struct("OwnedBase")
                .field("aref", &self.aref)
                .field("own", &self.own)
                .finish()
        }
    }
}

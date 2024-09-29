//! Owning array types.

use core::{fmt::Debug, marker::PhantomData};

use super::{
    arrayref::{RawRefBase, RefBase},
    Owner, Ref,
};

/// An array with its own shape and (possibly-shared) data.
/// 
/// The `ArrayBase` is used to jointly represent owned, immutably viewed, and
/// mutably viewed arrays, and is the primary interface for constructing arrays.
/// Since `ArrayBase` dereferences to [`RefBase`], it shares all of its invariants.
/// Its data is safe to read and, when either owned or mutably viewed, safe to
/// mutate.
/// 
/// The array is parameterized by `L` for its layout type, `R` for its reference
/// type, and `O` for its ownership type. `R` and `O` are separated in order to allow
/// the packing of both viewed and owned arrays into the same type; this makes implementing
/// functions for `ArrayBase` considerably easier.
/// 
/// # Ownership vs Views
/// When `O` implements [`Owner`], the array has (possibly shared) ownership over its data;
/// [`OwnedBase`] is an alias for this case. Although `O` could be any type that doesn't
/// implement `Owner`, it generally falls into one of two cases: [`ViewBase`] and [`ViewBaseMut`],
/// representing immutable and mutable views, respectively.
/// 
/// # Functionality and [`RefBase`]
/// Like the relationship between [`Vec<T>`] and `&[T]`, most functionality for arrays
/// is implemented on `RefBase` or [`RawRefBase`]. Implementors of new functions or traits
/// should try to implement them as far "down" this stack as possible, first preferring
/// `RawRefBase`, then `RefBase` (if data must be aligned and safe to dereference), and only
/// then on `ArrayBase` if access to the ownership is needed.
#[derive(Debug)]
pub struct ArrayBase<L, R, O>
where
    R: Ref,
{
    aref: RefBase<L, R>,
    own: O,
}

impl<L, R: Ref, O> ArrayBase<L, R, O> {
    pub unsafe fn new_unchecked(storage: R, layout: L, own: O) -> Self {
        Self {
            aref: RefBase::new_unchecked(storage, layout),
            own,
        }
    }

    pub unsafe fn with_layout<Y>(self, layout: Y) -> ArrayBase<Y, R, O> {
        ArrayBase {
            aref: self.aref.with_layout(layout),
            own: self.own,
        }
    }

    // pub unsafe fn with_storage(self, storage: R) -> Self {
    //     Self {
    //         aref: self.aref.with_storage(storage),
    //         own: self.own,
    //     }
    // }
}

/// An array view with data that may not be aligned or safe to dereference, and without lifetimes.
#[derive(Debug)]
pub struct RawArrayViewBase<L, R, I>
where
    R: Ref,
{
    rref: RawRefBase<L, R>,
    life: PhantomData<I>,
}

/// Base type for arrays with owning semantics.
pub type OwnedBase<L, O> = ArrayBase<L, <O as Owner>::Ref, O>;

impl<L, O: Owner> OwnedBase<L, O> {
    pub fn own(&self) -> &O {
        &self.own
    }
}

/// Base type for array views.
///
/// Views are like references; in fact, they're just wrappers for references.
/// The difference is that a reference's layout must be identical to the array
/// from which is has been derived; a view's layout may be different, representing
/// a segment, strided, or otherwise incomplete look at its parent array.
pub type ViewBase<'a, L, R> = ArrayBase<L, R, PhantomData<&'a <R as Ref>::Elem>>;

/// Base type for array views with mutable data.
///
/// All kinds of views can have their layout mutated. However, data mutation
/// is tracked separately via two different types.
pub type ViewBaseMut<'a, L, R> = ArrayBase<L, R, PhantomData<&'a mut <R as Ref>::Elem>>;

/// Base type for array views without lifetimes.
pub type RawViewBase<L, R> = RawArrayViewBase<L, R, *const <R as Ref>::Elem>;

/// Base type for array views without lifetimes, but with mutable data.
pub type RawViewBaseMut<L, R> = RawArrayViewBase<L, R, *mut <R as Ref>::Elem>;

/// A trait for arrays with mutable data that can be made unique.
///
/// Essentially all monomorphizations of [`ArrayBase`] should implement
/// `Uniqueable`; this applies even when the array type does not have
/// any data sharing capabilities.
///
/// There are already blanket implementations for `ViewBaseMut` and
/// `RawViewBaseMut`; as a result, any creation of these types
/// _must_ ensure that the underlying data is unique (a.k.a, unshared)
/// before creating these mutable views.
/// 
/// # Safety
/// `Uniqueable` is the trait that guarantees some of the invariants
/// promised by [`RawRefBase`] and [`RefBase`], as it is the trait that is
/// called when dereferencing [`ArrayBase`].
pub unsafe trait Uniqueable {
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
pub unsafe trait NdArray<R: Ref> {
    fn as_ptr(&self) -> *mut R::Elem;
}

mod array_impls {
    use core::ops::{Deref, DerefMut};

    use crate::core::{Owner, RawRefBase, Ref, RefBase};

    use super::{ArrayBase, RawArrayViewBase, RawViewBaseMut, Uniqueable, ViewBaseMut};

    impl<L, R: Ref, I> Deref for RawArrayViewBase<L, R, I> {
        type Target = RawRefBase<L, R>;

        fn deref(&self) -> &Self::Target {
            &self.rref
        }
    }

    impl<L, R: Ref> DerefMut for RawViewBaseMut<L, R> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.rref
        }
    }

    impl<L, R: Ref, O> Deref for ArrayBase<L, R, O> {
        type Target = RefBase<L, R>;

        fn deref(&self) -> &Self::Target {
            &self.aref
        }
    }

    impl<L, O: Owner> DerefMut for ArrayBase<L, O::Ref, O> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.aref
        }
    }

    unsafe impl<'a, L, R: Ref> Uniqueable for ViewBaseMut<'a, L, R> {
        fn try_ensure_unique(&mut self) {}

        fn try_is_unique(&self) -> Option<bool> {
            Some(true)
        }
    }

    unsafe impl<L, R: Ref> Uniqueable for RawViewBaseMut<L, R> {
        fn try_ensure_unique(&mut self) {}

        fn try_is_unique(&self) -> Option<bool> {
            None
        }
    }
}

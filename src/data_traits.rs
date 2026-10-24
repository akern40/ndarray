// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The data (inner representation) traits for ndarray

#[allow(unused_imports)]
use rawpointer::PointerExt;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::mem::MaybeUninit;

use crate::buffer::Buffer;
use crate::data_repr::DenseBuf;
use crate::storage::{Cow, Owned, RawView, RawViewMut, Shared, View, ViewMut};
use crate::{ArrayBase, Dimension};

/// Array representation trait.
///
/// For an array that meets the invariants of the `ArrayBase` type. This trait
/// does not imply any ownership or lifetime; pointers to elements in the array
/// may not be safe to dereference.
///
/// ***Note:*** `RawData` is not an extension interface at this point.
/// Traits in Rust can serve many different roles. This trait is public because
/// it is used as a bound on public methods.
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait RawData: Sized
{
    /// The array element type.
    type Elem;

    /// The buffer type: which format the elements are stored in. The handle
    /// type is `<Self::Buf as Buffer>::Handle`.
    type Buf: Buffer<Elem = Self::Elem>;

    #[doc(hidden)]
    fn _is_pointer_inbounds(&self, handle: <Self::Buf as Buffer>::Handle) -> bool;

    private_decl! {}
}

/// Array representation trait.
///
/// For an array with writable elements.
///
/// ***Internal trait, see `RawData`.***
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait RawDataMut: RawData
{
    /// If possible, ensures that the array has unique access to its data.
    ///
    /// The implementer must ensure that if the input is contiguous, then the
    /// output has the same strides as input.
    ///
    /// Additionally, if `Self` provides safe mutable access to array elements,
    /// then this method **must** panic or ensure that the data is unique.
    #[doc(hidden)]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension;

    /// If possible, returns whether the array has unique access to its data.
    ///
    /// If `Self` provides safe mutable access to array elements, then it
    /// **must** return `Some(_)`.
    #[doc(hidden)]
    fn try_is_unique(&mut self) -> Option<bool>;
}

/// Array representation trait.
///
/// An array representation that can be cloned.
///
/// ***Internal trait, see `RawData`.***
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait RawDataClone: RawData
{
    #[doc(hidden)]
    /// Unsafe because, `ptr` must point inside the current storage.
    unsafe fn clone_with_ptr(&self, ptr: <Self::Buf as Buffer>::Handle) -> (Self, <Self::Buf as Buffer>::Handle);

    #[doc(hidden)]
    unsafe fn clone_from_with_ptr(
        &mut self, other: &Self, ptr: <Self::Buf as Buffer>::Handle,
    ) -> <Self::Buf as Buffer>::Handle
    {
        let (data, ptr) = other.clone_with_ptr(ptr);
        *self = data;
        ptr
    }
}

/// Array representation trait.
///
/// For an array with elements that can be accessed with safe code.
///
/// ***Internal trait, see `RawData`.***
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait Data: RawData {}

/// Array representation trait.
///
/// For an array with writable elements that can be accessed with safe code.
///
/// ***Internal trait, see `Data`.***
//
// # For implementers
//
// If you implement the `DataMut` trait, you are guaranteeing that the
// `RawDataMut::try_ensure_unique` implementation always panics or ensures that
// the data is unique. You are also guaranteeing that `try_is_unique` always
// returns `Some(_)`.
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait DataMut: Data + RawDataMut
{
    /// Ensures that the array has unique access to its data.
    #[doc(hidden)]
    #[inline]
    fn ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
        Self::try_ensure_unique(self_)
    }

    /// Returns whether the array has unique access to its data.
    #[doc(hidden)]
    #[inline]
    #[allow(clippy::wrong_self_convention)] // mut needed for Arc types
    fn is_unique(&mut self) -> bool
    {
        self.try_is_unique().unwrap()
    }
}

/// Array representation trait.
///
/// A representation that can become uniquely owned, keeping its format
/// (a dense view becomes a dense array, a sparse view a sparse array).
///
/// ***Internal trait, see `Data`.***
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait IntoOwned: RawData
{
    /// The uniquely owned storage of this format.
    type Owned: DataMut<Elem = Self::Elem, Buf = Self::Buf>;

    #[doc(hidden)]
    #[allow(clippy::wrong_self_convention)]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Self::Owned, D>
    where
        Self::Elem: Clone,
        D: Dimension;

    #[doc(hidden)]
    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<ArrayBase<Self::Owned, D>, ArrayBase<Self, D>>
    where D: Dimension;
}

/// Array representation trait.
///
/// A representation that can become shared (copy-on-write), keeping its format.
///
/// ***Internal trait, see `Data`.***
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait IntoShared: Data
{
    /// The shared storage of this format.
    type Shared: DataShared<Elem = Self::Elem, Buf = Self::Buf>;

    #[doc(hidden)]
    #[allow(clippy::wrong_self_convention)]
    fn into_shared<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Self::Shared, D>
    where
        Self::Elem: Clone,
        D: Dimension;

    #[doc(hidden)]
    #[allow(clippy::wrong_self_convention)]
    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<Self::Shared, D>
    where
        Self::Elem: Clone,
        D: Dimension;
}

/// Array representation trait.
///
/// A representation which can be the owner of its data.
///
/// ***Internal trait, see `Data`.***
// The owned storage represents the ownership and allocation of the array's elements.
// The storage may be unique or shared ownership style; it must be an aliasable owner
// (permit aliasing pointers, such as our separate array head pointer).
//
// The array storage must be initially mutable - copy on write arrays may require copying for
// unsharing storage before mutating it. The initially allocated storage must be mutable so
// that it can be mutated directly - through .raw_view_mut_unchecked() - for initialization.
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait DataOwned:
    Data<Buf = DenseBuf<<Self as RawData>::Elem>> + IntoOwned<Owned = Owned<DenseBuf<<Self as RawData>::Elem>>>
{
    /// Corresponding owned data with MaybeUninit elements
    type MaybeUninit: DataOwned<Elem = MaybeUninit<Self::Elem>> + RawDataSubst<Self::Elem, Output = Self>;
    #[doc(hidden)]
    fn new(elements: Vec<Self::Elem>) -> Self;
}

/// Array representation trait.
///
/// A representation that is a lightweight view.
///
/// ***Internal trait, see `Data`.***
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait DataShared: Clone + Data + RawDataClone {}
unsafe impl<A> DataOwned for Owned<DenseBuf<A>>
{
    type MaybeUninit = Owned<DenseBuf<MaybeUninit<A>>>;

    fn new(elements: Vec<A>) -> Self
    {
        Owned(DenseBuf::from(elements))
    }
}

unsafe impl<A> DataOwned for Shared<DenseBuf<A>>
{
    type MaybeUninit = Shared<DenseBuf<MaybeUninit<A>>>;

    fn new(elements: Vec<A>) -> Self
    {
        Shared(Arc::new(DenseBuf::from(elements)))
    }
}

unsafe impl<'a, A> DataOwned for Cow<'a, DenseBuf<A>>
{
    type MaybeUninit = Cow<'a, DenseBuf<MaybeUninit<A>>>;

    fn new(elements: Vec<A>) -> Self
    {
        Cow::Owned(Owned(DenseBuf::from(elements)))
    }
}

/// Array representation trait.
///
/// TODO(design): revisit. Reinterprets a storage of `A` as the same storage of
/// `B` when the two have identical memory layout (`cast`, `assume_init`). In the
/// buffer/wrapper design this is a buffer-level `DenseBuf<A> -> DenseBuf<B>` with
/// the wrapper mapping generically; dense by nature either way.
///
/// The RawDataSubst trait maps the element type of array storage, while
/// keeping the same kind of storage.
///
/// For example, `RawDataSubst<B>` can map the type `OwnedRepr<A>` to `OwnedRepr<B>`.
pub trait RawDataSubst<A>: RawData<Buf = DenseBuf<<Self as RawData>::Elem>>
{
    /// The resulting array storage of the same kind but substituted element type
    type Output: RawData<Elem = A, Buf = DenseBuf<A>>;

    /// Unsafely translate the data representation from one element
    /// representation to another.
    ///
    /// ## Safety
    ///
    /// Caller must ensure the two types have the same representation.
    unsafe fn data_subst(self) -> Self::Output;
}

impl<A, B> RawDataSubst<B> for Owned<DenseBuf<A>>
{
    type Output = Owned<DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        Owned(self.0.data_subst())
    }
}

impl<A, B> RawDataSubst<B> for Shared<DenseBuf<A>>
{
    type Output = Shared<DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        Shared(Arc::from_raw(Arc::into_raw(self.0) as *const DenseBuf<B>))
    }
}

impl<A, B> RawDataSubst<B> for RawView<DenseBuf<A>>
{
    type Output = RawView<DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        RawView::new()
    }
}

impl<A, B> RawDataSubst<B> for RawViewMut<DenseBuf<A>>
{
    type Output = RawViewMut<DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        RawViewMut::new()
    }
}

impl<'a, A: 'a, B: 'a> RawDataSubst<B> for View<'a, DenseBuf<A>>
{
    type Output = View<'a, DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        View::new()
    }
}

impl<'a, A: 'a, B: 'a> RawDataSubst<B> for ViewMut<'a, DenseBuf<A>>
{
    type Output = ViewMut<'a, DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        ViewMut::new()
    }
}

impl<'a, A: 'a, B: 'a> RawDataSubst<B> for Cow<'a, DenseBuf<A>>
{
    type Output = Cow<'a, DenseBuf<B>>;

    unsafe fn data_subst(self) -> Self::Output
    {
        match self {
            Cow::View(view) => Cow::View(view.data_subst()),
            Cow::Owned(owned) => Cow::Owned(owned.data_subst()),
        }
    }
}

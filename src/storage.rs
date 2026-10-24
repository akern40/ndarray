//! Ownership wrappers: how a [`Buffer`] is held, written once for every format.
//!
//! * [`Owned<B>`]: uniquely owned.
//! * [`Shared<B>`]: `Arc`-shared, copy-on-write.
//! * [`View<'a, B>`] / [`ViewMut<'a, B>`]: borrowed.
//! * [`RawView<B>`] / [`RawViewMut<B>`]: borrowed without a lifetime; not readable.
//! * [`Cow<'a, B>`]: a view until written to, then owned.

use core::marker::PhantomData;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use crate::buffer::Buffer;
use crate::data_traits::{Data, DataMut, DataShared, IntoOwned, IntoShared, RawData, RawDataClone, RawDataMut};
use crate::handle::Handle;
use crate::{ArrayBase, Dimension};

/// Uniquely owned buffer.
#[derive(Debug)]
pub struct Owned<B>(pub(crate) B);

/// Shared buffer with copy-on-write.
#[derive(Debug)]
pub struct Shared<B>(pub(crate) Arc<B>);

/// Borrowed buffer, read-only.
pub struct View<'a, B>(PhantomData<&'a B>);

/// Borrowed buffer, read-write.
pub struct ViewMut<'a, B>(PhantomData<&'a mut B>);

/// Borrowed buffer without a lifetime; elements may not be read.
pub struct RawView<B>(PhantomData<*const B>);

/// Mutably borrowed buffer without a lifetime; elements may not be read.
pub struct RawViewMut<B>(PhantomData<*mut B>);

// Marker types: Copy/Clone regardless of `B`.
impl<B> Clone for View<'_, B>
{
    fn clone(&self) -> Self
    {
        *self
    }
}
impl<B> Copy for View<'_, B> {}
impl<B> Clone for RawView<B>
{
    fn clone(&self) -> Self
    {
        *self
    }
}
impl<B> Copy for RawView<B> {}
impl<B> Clone for RawViewMut<B>
{
    fn clone(&self) -> Self
    {
        *self
    }
}
impl<B> Copy for RawViewMut<B> {}

/// A view until first written to, then an owned buffer.
pub enum Cow<'a, B>
{
    /// Borrowed data.
    View(View<'a, B>),
    /// Owned data.
    Owned(Owned<B>),
}

/// A read-only view of any storage's buffer, without going through `Deref`
/// (whose target is an unnormalised projection inside generic impls).
fn view_of<B: Buffer, S: RawData<Buf = B>, D: Dimension>(a: &ArrayBase<S, D>) -> ArrayBase<View<'_, B>, D>
{
    unsafe { ArrayBase::from_data_ptr(View::new(), a.parts.handle).with_layout(a.parts.layout.clone()) }
}

impl<B> View<'_, B>
{
    #[inline(always)]
    pub(crate) const fn new() -> Self
    {
        View(PhantomData)
    }
}

impl<B> ViewMut<'_, B>
{
    #[inline(always)]
    pub(crate) const fn new() -> Self
    {
        ViewMut(PhantomData)
    }
}

impl<B> RawView<B>
{
    #[inline(always)]
    pub(crate) const fn new() -> Self
    {
        RawView(PhantomData)
    }
}

impl<B> RawViewMut<B>
{
    #[inline(always)]
    pub(crate) const fn new() -> Self
    {
        RawViewMut(PhantomData)
    }
}

impl<B> Cow<'_, B>
{
    /// Returns `true` iff the data is the `View` variant.
    pub fn is_view(&self) -> bool
    {
        matches!(self, Cow::View(_))
    }

    /// Returns `true` iff the data is the `Owned` variant.
    pub fn is_owned(&self) -> bool
    {
        matches!(self, Cow::Owned(_))
    }
}

impl<B: Clone> Clone for Owned<B>
{
    fn clone(&self) -> Self
    {
        Owned(self.0.clone())
    }

    fn clone_from(&mut self, other: &Self)
    {
        self.0.clone_from(&other.0)
    }
}

impl<B> Clone for Shared<B>
{
    fn clone(&self) -> Self
    {
        Shared(self.0.clone())
    }
}

impl<B> core::ops::Deref for Owned<B>
{
    type Target = B;
    fn deref(&self) -> &B
    {
        &self.0
    }
}

impl<B> core::ops::DerefMut for Owned<B>
{
    fn deref_mut(&mut self) -> &mut B
    {
        &mut self.0
    }
}

// ---- Owned<B> ----

unsafe impl<B: Buffer> RawData for Owned<B>
{
    type Elem = B::Elem;
    type Buf = B;

    fn _is_pointer_inbounds(&self, handle: B::Handle) -> bool
    {
        self.0.is_inbounds(handle)
    }

    private_impl! {}
}

unsafe impl<B: Buffer> RawDataMut for Owned<B>
{
    #[inline]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool>
    {
        Some(true)
    }
}

unsafe impl<B: Buffer> Data for Owned<B> {}
unsafe impl<B: Buffer> DataMut for Owned<B> {}

unsafe impl<B: Buffer> RawDataClone for Owned<B>
where B::Elem: Clone
{
    unsafe fn clone_with_ptr(&self, handle: B::Handle) -> (Self, B::Handle)
    {
        let off = self.0.offset_of(handle);
        let u = self.0.clone_buf();
        let h = u.origin().offset(off);
        (Owned(u), h)
    }

    unsafe fn clone_from_with_ptr(&mut self, other: &Self, handle: B::Handle) -> B::Handle
    {
        let off = other.0.offset_of(handle);
        self.0.clone_buf_from(&other.0);
        self.0.origin().offset(off)
    }
}

unsafe impl<B: Buffer> IntoOwned for Owned<B>
{
    type Owned = Self;

    #[inline]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Self, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        self_
    }

    #[inline]
    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<ArrayBase<Self, D>, ArrayBase<Self, D>>
    where D: Dimension
    {
        Ok(self_)
    }
}

unsafe impl<B: Buffer> IntoShared for Owned<B>
{
    type Shared = Shared<B>;

    fn into_shared<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        // Handles survive a move of the buffer value (Buffer's contract).
        let parts = self_.parts;
        unsafe { ArrayBase::from_data_ptr(Shared(Arc::new(self_.data.0)), parts.handle).with_layout(parts.layout) }
    }

    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(B::collect(view_of(self_)))
    }
}

// ---- Shared<B> ----

unsafe impl<B: Buffer> RawData for Shared<B>
{
    type Elem = B::Elem;
    type Buf = B;

    fn _is_pointer_inbounds(&self, handle: B::Handle) -> bool
    {
        self.0.is_inbounds(handle)
    }

    private_impl! {}
}

// NOTE: Copy on write
unsafe impl<B: Buffer> RawDataMut for Shared<B>
where B::Elem: Clone
{
    fn try_ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
        if Arc::get_mut(&mut self_.data.0).is_some() {
            return;
        }
        if self_.parts.layout.dim.size() <= self_.data.0.len() / 2 {
            // Clone only the visible elements if the current view is less than
            // half of backing data.
            *self_ = <Owned<B> as IntoShared>::into_shared(B::collect(view_of(self_)));
            return;
        }
        let off = self_.data.0.offset_of(self_.parts.handle);
        // `Arc::make_mut` without `B: Clone`: the buffer is clonable exactly
        // when its elements are.
        self_.data.0 = Arc::new(self_.data.0.clone_buf());
        unsafe {
            self_.parts.handle = self_.data.0.origin().offset(off);
        }
    }

    fn try_is_unique(&mut self) -> Option<bool>
    {
        Some(Arc::get_mut(&mut self.0).is_some())
    }
}

unsafe impl<B: Buffer> Data for Shared<B> {}
unsafe impl<B: Buffer> DataMut for Shared<B> where B::Elem: Clone {}

unsafe impl<B: Buffer> RawDataClone for Shared<B>
{
    unsafe fn clone_with_ptr(&self, handle: B::Handle) -> (Self, B::Handle)
    {
        // handle is preserved: the buffer is shared, not copied
        (self.clone(), handle)
    }
}

unsafe impl<B: Buffer> DataShared for Shared<B> {}

unsafe impl<B: Buffer> IntoOwned for Shared<B>
{
    type Owned = Owned<B>;

    fn into_owned<D>(mut self_: ArrayBase<Self, D>) -> ArrayBase<Owned<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        Self::ensure_unique(&mut self_);
        let parts = self_.parts;
        let buf = Arc::try_unwrap(self_.data.0).ok().unwrap();
        unsafe { ArrayBase::from_data_ptr(Owned(buf), parts.handle).with_layout(parts.layout) }
    }

    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<ArrayBase<Owned<B>, D>, ArrayBase<Self, D>>
    where D: Dimension
    {
        let parts = self_.parts;
        match Arc::try_unwrap(self_.data.0) {
            Ok(buf) => unsafe { Ok(ArrayBase::from_data_ptr(Owned(buf), parts.handle).with_layout(parts.layout)) },
            Err(arc) => unsafe { Err(ArrayBase::from_data_ptr(Shared(arc), parts.handle).with_layout(parts.layout)) },
        }
    }
}

unsafe impl<B: Buffer> IntoShared for Shared<B>
{
    type Shared = Self;

    #[inline]
    fn into_shared<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Self, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        self_
    }

    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<Self, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        // shares the buffer; no element is copied
        self_.clone()
    }
}

// ---- View<'a, B> ----

unsafe impl<B: Buffer> RawData for View<'_, B>
{
    type Elem = B::Elem;
    type Buf = B;

    #[inline(always)]
    fn _is_pointer_inbounds(&self, _handle: B::Handle) -> bool
    {
        true
    }

    private_impl! {}
}

unsafe impl<B: Buffer> Data for View<'_, B> {}

unsafe impl<B: Buffer> RawDataClone for View<'_, B>
{
    unsafe fn clone_with_ptr(&self, handle: B::Handle) -> (Self, B::Handle)
    {
        (*self, handle)
    }
}

unsafe impl<B: Buffer> DataShared for View<'_, B> {}

unsafe impl<B: Buffer> IntoOwned for View<'_, B>
{
    type Owned = Owned<B>;

    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Owned<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        B::collect(view_of(&self_))
    }

    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<ArrayBase<Owned<B>, D>, ArrayBase<Self, D>>
    where D: Dimension
    {
        Err(self_)
    }
}

unsafe impl<B: Buffer> IntoShared for View<'_, B>
{
    type Shared = Shared<B>;

    fn into_shared<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(B::collect(view_of(&self_)))
    }

    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(B::collect(view_of(self_)))
    }
}

// ---- ViewMut<'a, B> ----

unsafe impl<B: Buffer> RawData for ViewMut<'_, B>
{
    type Elem = B::Elem;
    type Buf = B;

    #[inline(always)]
    fn _is_pointer_inbounds(&self, _handle: B::Handle) -> bool
    {
        true
    }

    private_impl! {}
}

unsafe impl<B: Buffer> RawDataMut for ViewMut<'_, B>
{
    #[inline]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool>
    {
        Some(true)
    }
}

unsafe impl<B: Buffer> Data for ViewMut<'_, B> {}
unsafe impl<B: Buffer> DataMut for ViewMut<'_, B> {}

unsafe impl<B: Buffer> IntoOwned for ViewMut<'_, B>
{
    type Owned = Owned<B>;

    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Owned<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        B::collect(view_of(&self_))
    }

    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<ArrayBase<Owned<B>, D>, ArrayBase<Self, D>>
    where D: Dimension
    {
        Err(self_)
    }
}

unsafe impl<B: Buffer> IntoShared for ViewMut<'_, B>
{
    type Shared = Shared<B>;

    fn into_shared<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(B::collect(view_of(&self_)))
    }

    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(B::collect(view_of(self_)))
    }
}

// ---- RawView<B> / RawViewMut<B> ----

unsafe impl<B: Buffer> RawData for RawView<B>
{
    type Elem = B::Elem;
    type Buf = B;

    #[inline(always)]
    fn _is_pointer_inbounds(&self, _handle: B::Handle) -> bool
    {
        true
    }

    private_impl! {}
}

unsafe impl<B: Buffer> RawDataClone for RawView<B>
{
    unsafe fn clone_with_ptr(&self, handle: B::Handle) -> (Self, B::Handle)
    {
        (*self, handle)
    }
}

unsafe impl<B: Buffer> RawData for RawViewMut<B>
{
    type Elem = B::Elem;
    type Buf = B;

    #[inline(always)]
    fn _is_pointer_inbounds(&self, _handle: B::Handle) -> bool
    {
        true
    }

    private_impl! {}
}

unsafe impl<B: Buffer> RawDataMut for RawViewMut<B>
{
    #[inline]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool>
    {
        Some(true)
    }
}

unsafe impl<B: Buffer> RawDataClone for RawViewMut<B>
{
    unsafe fn clone_with_ptr(&self, handle: B::Handle) -> (Self, B::Handle)
    {
        (*self, handle)
    }
}

// ---- Cow<'a, B> ----

unsafe impl<B: Buffer> RawData for Cow<'_, B>
{
    type Elem = B::Elem;
    type Buf = B;

    #[inline]
    fn _is_pointer_inbounds(&self, handle: B::Handle) -> bool
    {
        match self {
            Cow::View(view) => view._is_pointer_inbounds(handle),
            Cow::Owned(data) => data._is_pointer_inbounds(handle),
        }
    }

    private_impl! {}
}

unsafe impl<B: Buffer> RawDataMut for Cow<'_, B>
where B::Elem: Clone
{
    #[inline]
    fn try_ensure_unique<D>(array: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
        if let Cow::View(_) = array.data {
            let owned = B::collect(view_of(array));
            array.data = Cow::Owned(owned.data);
            array.parts.handle = owned.parts.handle;
            array.parts.layout = owned.parts.layout;
        }
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool>
    {
        Some(self.is_owned())
    }
}

unsafe impl<B: Buffer> RawDataClone for Cow<'_, B>
where B::Elem: Clone
{
    unsafe fn clone_with_ptr(&self, handle: B::Handle) -> (Self, B::Handle)
    {
        match self {
            Cow::View(view) => {
                let (new_view, handle) = view.clone_with_ptr(handle);
                (Cow::View(new_view), handle)
            }
            Cow::Owned(data) => {
                let (new_data, handle) = data.clone_with_ptr(handle);
                (Cow::Owned(new_data), handle)
            }
        }
    }

    unsafe fn clone_from_with_ptr(&mut self, other: &Self, handle: B::Handle) -> B::Handle
    {
        match (&mut *self, other) {
            (Cow::View(self_), Cow::View(other)) => self_.clone_from_with_ptr(other, handle),
            (Cow::Owned(self_), Cow::Owned(other)) => self_.clone_from_with_ptr(other, handle),
            (_, Cow::Owned(other)) => {
                let (cloned, handle) = other.clone_with_ptr(handle);
                *self = Cow::Owned(cloned);
                handle
            }
            (_, Cow::View(other)) => {
                let (cloned, handle) = other.clone_with_ptr(handle);
                *self = Cow::View(cloned);
                handle
            }
        }
    }
}

unsafe impl<B: Buffer> Data for Cow<'_, B> {}
unsafe impl<B: Buffer> DataMut for Cow<'_, B> where B::Elem: Clone {}

unsafe impl<B: Buffer> IntoOwned for Cow<'_, B>
{
    type Owned = Owned<B>;

    #[inline]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Owned<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        let parts = self_.parts;
        match self_.data {
            Cow::View(_) =>
                B::collect(unsafe { ArrayBase::from_data_ptr(View::new(), parts.handle).with_layout(parts.layout) }),
            Cow::Owned(data) => unsafe { ArrayBase::from_data_ptr(data, parts.handle).with_layout(parts.layout) },
        }
    }

    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<ArrayBase<Owned<B>, D>, ArrayBase<Self, D>>
    where D: Dimension
    {
        match self_.data {
            Cow::View(_) => Err(self_),
            Cow::Owned(data) => {
                let parts = self_.parts;
                unsafe { Ok(ArrayBase::from_data_ptr(data, parts.handle).with_layout(parts.layout)) }
            }
        }
    }
}

unsafe impl<B: Buffer> IntoShared for Cow<'_, B>
{
    type Shared = Shared<B>;

    fn into_shared<D>(self_: ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(<Self as IntoOwned>::into_owned(self_))
    }

    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<Shared<B>, D>
    where
        B::Elem: Clone,
        D: Dimension,
    {
        <Owned<B> as IntoShared>::into_shared(B::collect(view_of(self_)))
    }
}

//! Spike: a second handle family. Coordinate-sparse storage with a fill value,
//! sharing `StridedLayout<D>` (and every layout operation) with the dense family.

use std::fmt;
use std::ptr::NonNull;

use crate::data_traits::{Data, DataMut, RawData, RawDataClone, RawDataMut};
use crate::dimension::IntoDimension;
use crate::handle::Handle;
use crate::{ArrayBase, Dimension};

/// The buffer behind a sparse array: a fill value plus `(offset, value)` entries sorted by offset.
pub struct SparseBuf<A>
{
    span: usize,
    fill: A,
    entries: Vec<(usize, A)>,
}

/// Handle into a `SparseBuf`: the buffer plus the logical offset of element `[0, .., 0]`.
pub struct SparseHandle<A>
{
    buf: NonNull<SparseBuf<A>>,
    base: isize,
}

impl<A> Clone for SparseHandle<A>
{
    fn clone(&self) -> Self
    {
        *self
    }
}
impl<A> Copy for SparseHandle<A> {}

impl<A> fmt::Debug for SparseHandle<A>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        f.debug_struct("SparseHandle")
            .field("buf", &self.buf)
            .field("base", &self.base)
            .finish()
    }
}

/// Handle for the sparse family: `get` is a binary search, falling back to the fill value;
/// `get_mut` inserts the fill value if the slot is absent and returns a reference to the entry.
unsafe impl<A: Clone> Handle for SparseHandle<A>
{
    type Elem = A;

    unsafe fn offset(self, by: isize) -> SparseHandle<A>
    {
        SparseHandle {
            buf: self.buf,
            base: self.base + by,
        }
    }

    unsafe fn get<'a>(self, offset: isize) -> &'a A
    {
        let buf = &*self.buf.as_ptr();
        let o = (self.base + offset) as usize;
        match buf.entries.binary_search_by_key(&o, |e| e.0) {
            Ok(i) => &buf.entries[i].1,
            Err(_) => &buf.fill,
        }
    }

    unsafe fn get_mut<'a>(self, offset: isize) -> &'a mut A
    {
        let buf = &mut *self.buf.as_ptr();
        let o = (self.base + offset) as usize;
        let i = match buf.entries.binary_search_by_key(&o, |e| e.0) {
            Ok(i) => i,
            Err(i) => {
                buf.entries.insert(i, (o, buf.fill.clone()));
                i
            }
        };
        &mut buf.entries[i].1
    }
}

/// Owning sparse storage: a boxed `SparseBuf` (boxed so the handle survives moves of the array).
pub struct OwnedSparse<A>(NonNull<SparseBuf<A>>);

impl<A> OwnedSparse<A>
{
    fn new(buf: SparseBuf<A>) -> (Self, SparseHandle<A>)
    {
        let ptr = NonNull::from(Box::leak(Box::new(buf)));
        (OwnedSparse(ptr), SparseHandle { buf: ptr, base: 0 })
    }

    fn buf(&self) -> &SparseBuf<A>
    {
        unsafe { self.0.as_ref() }
    }
}

impl<A> Drop for OwnedSparse<A>
{
    fn drop(&mut self)
    {
        unsafe { drop(Box::from_raw(self.0.as_ptr())) }
    }
}

unsafe impl<A: Send> Send for OwnedSparse<A> {}
unsafe impl<A: Sync> Sync for OwnedSparse<A> {}

unsafe impl<A: Clone> RawData for OwnedSparse<A>
{
    type Elem = A;
    type Handle = SparseHandle<A>;

    fn _is_pointer_inbounds(&self, handle: SparseHandle<A>) -> bool
    {
        handle.buf == self.0 && handle.base >= 0 && handle.base as usize <= self.buf().span
    }

    private_impl! {}
}

unsafe impl<A: Clone> RawDataMut for OwnedSparse<A>
{
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    fn try_is_unique(&mut self) -> Option<bool>
    {
        Some(true)
    }
}

unsafe impl<A: Clone> Data for OwnedSparse<A>
{
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> crate::Array<A, D>
    where
        A: Clone,
        D: Dimension,
    {
        self_.view().into_owned()
    }

    fn try_into_owned_nocopy<D>(self_: ArrayBase<Self, D>) -> Result<crate::Array<A, D>, ArrayBase<Self, D>>
    where D: Dimension
    {
        Err(self_)
    }
}

unsafe impl<A: Clone> DataMut for OwnedSparse<A> {}

unsafe impl<A: Clone> RawDataClone for OwnedSparse<A>
{
    unsafe fn clone_with_ptr(&self, handle: SparseHandle<A>) -> (Self, SparseHandle<A>)
    {
        let buf = self.buf();
        let (data, fresh) = OwnedSparse::new(SparseBuf {
            span: buf.span,
            fill: buf.fill.clone(),
            entries: buf.entries.clone(),
        });
        (data, SparseHandle {
            buf: fresh.buf,
            base: handle.base,
        })
    }
}

/// An owned sparse array with a fill value.
pub type SparseArray<A, D> = ArrayBase<OwnedSparse<A>, D>;

impl<A: Clone, D: Dimension> SparseArray<A, D>
{
    /// Create a sparse array of the given shape where every element is `fill` and nothing is stored.
    pub fn from_elem<Sh>(shape: Sh, fill: A) -> Self
    where Sh: IntoDimension<Dim = D>
    {
        let dim = shape.into_dimension();
        let span = dim
            .size_checked()
            .expect("SparseArray::from_elem: shape overflow");
        let strides = dim.default_strides();
        let (data, handle) = OwnedSparse::new(SparseBuf {
            span,
            fill,
            entries: Vec::new(),
        });
        unsafe { ArrayBase::from_data_ptr(data, handle).with_strides_dim(strides, dim) }
    }

    /// Number of explicitly stored elements.
    pub fn nnz(&self) -> usize
    {
        self.data.buf().entries.len()
    }

    /// The fill value.
    pub fn fill_value(&self) -> &A
    {
        &self.data.buf().fill
    }
}

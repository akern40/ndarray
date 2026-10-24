//! Spike: a second handle family. Coordinate-sparse storage with a fill value,
//! sharing `StridedLayout<D>` (and every layout operation) with the dense family.

use crate::StridedLayout;
use std::fmt;
use std::ptr::NonNull;

use crate::buffer::Buffer;
use crate::dimension::IntoDimension;
use crate::handle::Handle;
use crate::storage::{Owned, View};
use crate::{ArrayBase, Dimension, RawData};

/// The buffer behind a sparse array: a fill value plus `(offset, value)` entries sorted by offset.
pub struct SparseBuf<A>(Box<SparseInner<A>>);

/// Heap-allocated so handles stay valid when the `SparseBuf` value moves.
#[derive(Debug, Clone)]
pub struct SparseInner<A>
{
    span: usize,
    fill: A,
    entries: Vec<(usize, A)>,
}

impl<A: Clone> Clone for SparseBuf<A>
{
    fn clone(&self) -> Self
    {
        SparseBuf(self.0.clone())
    }
}

/// Handle into a `SparseBuf`: the buffer plus the logical offset of element `[0, .., 0]`.
pub struct SparseHandle<A>
{
    buf: NonNull<SparseInner<A>>,
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
unsafe impl<A: Clone> Buffer for SparseBuf<A>
{
    type Elem = A;
    type Handle = SparseHandle<A>;

    fn origin(&self) -> SparseHandle<A>
    {
        SparseHandle {
            buf: NonNull::from(&*self.0),
            base: 0,
        }
    }

    fn offset_of(&self, handle: SparseHandle<A>) -> isize
    {
        handle.base
    }

    fn is_inbounds(&self, handle: SparseHandle<A>) -> bool
    {
        handle.buf == NonNull::from(&*self.0) && handle.base >= 0 && handle.base as usize <= self.0.span
    }

    fn len(&self) -> usize
    {
        self.0.span
    }

    fn clone_buf(&self) -> Self
    {
        self.clone()
    }

    fn collect<D: Dimension>(
        view: ArrayBase<View<'_, Self>, StridedLayout<D>>,
    ) -> ArrayBase<Owned<Self>, StridedLayout<D>>
    {
        // Standard layout; keep only the slots that have an explicit entry.
        let fill = unsafe { &*view.parts.handle.buf.as_ptr() }.fill.clone();
        let mut entries = Vec::new();
        let span = view.len();
        let dim = view.raw_dim();
        for (k, h) in view.into_base_iter().enumerate() {
            let inner = unsafe { &*h.buf.as_ptr() };
            if let Ok(i) = inner
                .entries
                .binary_search_by_key(&(h.base as usize), |e| e.0)
            {
                entries.push((k, inner.entries[i].1.clone()));
            }
        }
        let buf = SparseBuf(Box::new(SparseInner { span, fill, entries }));
        let handle = buf.origin();
        unsafe { ArrayBase::from_data_ptr(Owned(buf), handle).with_strides_dim(dim.default_strides(), dim) }
    }
}

/// An array in the sparse format, uniquely owned.
pub type SparseArray<A, D> = ArrayBase<Owned<SparseBuf<A>>, StridedLayout<D>>;

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
        let buf = SparseBuf(Box::new(SparseInner {
            span,
            fill,
            entries: Vec::new(),
        }));
        let handle = buf.origin();
        unsafe { ArrayBase::from_data_ptr(Owned(buf), handle).with_strides_dim(strides, dim) }
    }
}

/// Any storage of the sparse format (owned, shared, view, ...).
impl<A: Clone, S: RawData<Buf = SparseBuf<A>>, D: Dimension> ArrayBase<S, StridedLayout<D>>
{
    fn inner(&self) -> &SparseInner<A>
    {
        unsafe { &*self.parts.handle.buf.as_ptr() }
    }

    /// Number of explicitly stored elements in the whole buffer.
    pub fn nnz(&self) -> usize
    {
        self.inner().entries.len()
    }

    /// The fill value.
    pub fn fill_value(&self) -> &A
    {
        &self.inner().fill
    }
}

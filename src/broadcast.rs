use crate::{ArrayBase, ArrayRef, ArrayView, ArrayView1, Data, Dimension, IntoDimension, Ix0, Ix1, ShapeBuilder};

/// A trait for broadcastable arguments to array functions.
///
///
pub trait Broadcastable<A, const ARR: bool>
{
    type PseudoDim: Dimension;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        A: Clone;

    fn broadcast_dim(&self) -> Self::PseudoDim;

    private_decl!();
}

impl<A> Broadcastable<A, false> for A
{
    type PseudoDim = Ix0;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        A: Clone,
    {
        let shape = shape.into_dimension();
        let ndim = shape.ndim();
        Some(
            unsafe {
                ArrayView::from_shape_ptr(
                    shape
                        .into_pattern()
                        .strides(E::Dim::zeros(ndim).into_pattern()),
                    self as *const A,
                )
            }
            .into(),
        )
    }

    fn broadcast_dim(&self) -> Self::PseudoDim
    {
        Ix0()
    }

    private_impl!();
}

impl<A, D> Broadcastable<A, true> for ArrayRef<A, D>
where D: Dimension
{
    type PseudoDim = D;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        A: Clone,
    {
        self.broadcast(shape).map(Into::into)
    }

    fn broadcast_dim(&self) -> Self::PseudoDim
    {
        self.raw_dim()
    }

    private_impl!();
}

impl<S, A, D> Broadcastable<A, true> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    type PseudoDim = D;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        A: Clone,
    {
        self.broadcast(shape).map(Into::into)
    }

    fn broadcast_dim(&self) -> Self::PseudoDim
    {
        self.raw_dim()
    }

    private_impl!();
}

impl<'a, A, Slice: ?Sized> Broadcastable<A, true> for Slice
where Slice: AsRef<[A]>
{
    type PseudoDim = Ix1;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        A: Clone,
    {
        ArrayView1::from(self.as_ref()).broadcast(shape).map(|v| {
            ArrayView::from_shape(v.raw_dim(), self.as_ref())
                .expect("Strides should be fine if they come from broadcast")
        })
    }

    fn broadcast_dim(&self) -> Self::PseudoDim
    {
        Ix1(self.as_ref().len())
    }

    private_impl!();
}

#[cfg(test)]
mod tests
{
    use crate::{array, broadcast::Broadcastable, Array1, Ix0, Ix1};

    #[test]
    fn test_elem()
    {
        let a = 1.0;
        assert_eq!(a.broadcast_dim(), Ix0());
        assert!(a.maybe_broadcast((10,)).is_some());
        assert_eq!(a.maybe_broadcast((10,)).unwrap(), Array1::from_elem((10,), 1.0));
        assert!(a.maybe_broadcast(()).is_some());
    }
}

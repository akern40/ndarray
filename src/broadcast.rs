use crate::{ArrayBase, ArrayRef, ArrayView, Data, Dimension, Ix0, ShapeBuilder};

/// A trait for broadcastable arguments to array functions.
///
///
pub trait Broadcastable<A, const ARR: bool>
{
    type PseudoDim: Dimension;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E>>
    where
        E: Dimension,
        A: Clone;

    fn broadcast_dim(&self) -> Self::PseudoDim;

    private_decl!();
}

impl<A> Broadcastable<A, false> for A
{
    type PseudoDim = Ix0;

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E>>
    where
        E: Dimension,
        A: Clone,
    {
        let ndim = shape.ndim();
        Some(
            unsafe {
                ArrayView::from_shape_ptr(shape.into_pattern().strides(E::zeros(ndim).into_pattern()), self as *const A)
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

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E>>
    where
        E: Dimension,
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

    fn maybe_broadcast<E>(&self, shape: E) -> Option<ArrayView<'_, A, E>>
    where
        E: Dimension,
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

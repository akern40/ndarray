// Copyright 2021 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::{imp_prelude::*, ArrayPartsSized};

// internal "builder-like" methods
impl<A, S> ArrayBase<S, Ix1>
where S: RawData<Elem = A>
{
    /// Create an (initially) empty one-dimensional array from the given data and array head
    /// pointer
    ///
    /// ## Safety
    ///
    /// The caller must ensure that the data storage and pointer is valid.
    ///
    /// See ArrayView::from_shape_ptr for general pointer validity documentation.
    #[inline]
    pub(crate) unsafe fn from_data_ptr(data: S, ptr: <S::Buf as Buffer>::Handle) -> Self
    {
        let array = ArrayBase {
            data,
            parts: ArrayPartsSized::new(ptr, Ix1(0), Ix1(1)),
        };
        debug_assert!(array.pointer_is_inbounds());
        array
    }
}

// internal "builder-like" methods
impl<A, S, D> ArrayBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    /// Set strides and dimension of the array to the new values
    ///
    /// The argument order with strides before dimensions is used because strides are often
    /// computed as derived from the dimension.
    ///
    /// ## Safety
    ///
    /// The caller needs to ensure that the new strides and dimensions are correct
    /// for the array data.
    #[inline]
    pub(crate) unsafe fn with_strides_dim<E>(self, strides: E, dim: E) -> ArrayBase<S, E>
    where E: Dimension
    {
        debug_assert_eq!(strides.ndim(), dim.ndim());
        ArrayBase {
            data: self.data,
            parts: ArrayPartsSized::new(self.parts.handle, dim, strides),
        }
    }

    /// Set the layout of the array, keeping storage and handle.
    ///
    /// ## Safety
    ///
    /// Every offset the new layout can produce must be in bounds of the
    /// array data, and the layout must be unique if the array is mutable.
    #[inline]
    pub(crate) unsafe fn with_layout<E>(self, layout: StridedLayout<E>) -> ArrayBase<S, E>
    where E: Dimension
    {
        self.with_strides_dim(layout.strides, layout.dim)
    }
}

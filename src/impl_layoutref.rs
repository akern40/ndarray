//! Methods for [`crate::LayoutRef`] that don't have to do with the interconnection of types.

use alloc::borrow::Cow;

use crate::{
    layout::{Layout, Strided},
    LayoutRef,
};

impl<A, L> LayoutRef<A, L> where L: Layout {}

impl<A, L> LayoutRef<A, L>
where L: Strided
{
    pub fn raw_strides(&self) -> Cow<'_, L::Strides>
    {
        self.layout.strides()
    }
}

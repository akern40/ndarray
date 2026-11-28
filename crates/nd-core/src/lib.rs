#![no_std]

mod array;
mod storage;
mod view;

pub use array::{ArrayAxis, ArrayBase};
pub use view::{SliceView, SliceViewAxis, SliceViewMut, SliceViewMutAxis};

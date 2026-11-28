#![no_std]

extern crate alloc;

mod paged;
mod position_space;
mod slice;

pub use paged::PagedStorage;
pub use position_space::PositionSpace;
pub use slice::{SliceStorage, SliceStorageMut};
